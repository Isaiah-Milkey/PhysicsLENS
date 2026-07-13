"""
Stage 3 · Specialist — Gravity
--------------------------------
Verifies that airborne objects obey "constant downward acceleration ⇒
parabolic trajectory", reusing the kinematics earlier stages already computed
(design: docs/superpowers/specs/2026-07-09-gravity-specialist-design.md).

Evidence in (bus first, standalone fallback):
  * s2_trajectory_extractor — smoothed positions/velocities/speed per track,
    contact events, explained impacts, Stage-2 anomaly flags (corroboration).
  * s2_object_tracker       — subject labels; SAM3 masks fix the coordinate
    frame height and align frame decode for VLM crops.
  * s2_hypothesis_generator — a ranked gravity hypothesis (t_window + reason)
    prioritizes which candidates get the bounded VLM budget.
  * Fallback: tools.tracking.get_tracks (shared cached LK tracker) + local
    gradient kinematics — degraded but standalone.

Detection vs explanation (same split as the other specialists):
numeric fits DETECT, the VLM EXPLAINS/CONFIRMS. Per track, free-flight
segments (moving, no contact/impact nearby, off the frame bottom) each get a
least-squares parabola fit y(t) = c0 + c1·t + c2·t²  (y-down image coords, so
normal gravity = a_fit = 2·c2 > 0). All checks are scale-free:

  non_parabolic_fall  — low R², large residual RMS vs vertical span, or a
                        residual step (teleport/stair artifacts)
  inconsistent_accel  — first-half vs second-half fitted accel disagree
  anti_gravity        — clearly-curved upward acceleration while airborne
                        (weak upward drift is a float candidate, not this)
  float               — no measurable curvature over ≥ hover_min_s airborne,
                        hover in place (named tracks only), or apex hang-time
  unequal_fall        — two simultaneously-airborne objects fall with clearly
                        different accelerations (Galileo)
  apex_asymmetry      — curvature before vs after a throw's apex disagrees
                        (each leg floored to measurable curvature first)

EVERY violation is VLM-confirmable (confirmed=None until judged): drag,
thrust, buoyancy and off-screen support can excuse any of them — a feather's
non-parabolic fall is real physics. Float candidates only count once the VLM
confirms; the other types are reported unverified without credentials, with
overall severity capped at 70 until something is actually confirmed.

Optional absolute-g: if the user supplies px_per_meter, g_est = a_fit / scale
is reported as a clearly-labeled *estimate* — it never drives severity.

Known limits (by design): camera motion is warned about and severity-capped
(coherent multi-track velocities), not compensated; hover detection needs
named object tracks (LK corner clusters park mid-frame constantly); segments
under ~5 samples are skipped.
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

GRAVITY_PROMPT = (
    "The image shows the SAME tracked object ({label}) at three moments of one "
    "video — its recent path is drawn on each tile and timestamps are printed. "
    "An automated ballistic-motion fit flagged this window as possibly "
    "violating gravity: {finding}.\n"
    "FIRST work out what the object actually is and its physical properties: "
    "material state (rigid solid, deformable, liquid, gas, granular), "
    "apparent weight/density, and how strongly air drag or buoyancy acts on "
    "it. Only THEN judge the finding against what gravity demands OF THAT "
    "KIND of object: dense rigid solids in free fall follow a clean parabola; "
    "very light high-drag objects sink slowly at near-constant speed; "
    "anything less dense than the air or water around it rises legitimately; "
    "a pouring liquid stream keeps a stationary shape while liquid flows "
    "through it (the shape hanging still is NOT floating); impacts throw "
    "droplets and fragments upward.\n"
    "Frames are sparsely sampled and the camera may move, so judge the "
    "object's OWN motion. Self-propelled or externally driven motion is also "
    "not a gravity violation: flying animals and machines, a person or "
    "animal jumping/climbing/being pushed, swinging or attached objects, and "
    "objects resting on or held by something possibly out of view.\n"
    "A violation requires BOTH: the object is genuinely UNSUPPORTED, and its "
    "motion is physically impossible FOR ITS IDENTIFIED PROPERTIES.\n"
    "Reply with ONLY strict JSON, fields IN THIS ORDER — do your reasoning in "
    "the early fields BEFORE committing to the verdict fields: "
    '{{"object": "<what it is>", '
    '"properties": "<material state · apparent weight · drag/buoyancy>", '
    '"reasoning": "<up to 3 sentences applying gravity to those properties>", '
    '"support": "unsupported"|"supported"|"self_propelled"|"unclear", '
    '"gravity_ok": true|false, "confidence": <0..1>, '
    '"explanation": "<one-sentence verdict>"}}'
)

VIOLATION_TYPES = ("non_parabolic_fall", "inconsistent_accel", "anti_gravity",
                   "float", "unequal_fall", "apex_asymmetry")

# Numeric-core defaults (settings override the first two; the rest are fixed
# thresholds documented in the spec).
DEFAULTS = {
    "min_airborne_s":   0.4,    # min free-flight duration worth fitting
    "equiv_tolerance":  1.5,    # Galileo: max plausible fall-accel ratio
    "min_samples":      5,      # min samples per segment
    "r2_min":           0.90,   # parabola-quality floor
    "accel_ratio_max":  2.0,    # half-vs-half / apex curvature ratio limit
    "floor_frac":       0.90,   # centroid below this frame fraction = grounded
    "hover_min_s":      0.5,    # min duration for float/hover candidates
}


def _sev_color(sev: float) -> str:
    return "#E24B4A" if sev > 60 else "#EF9F27" if sev > 30 else "#4CAF50"


def _jpeg_event(img: np.ndarray, caption: str, quality: int = 90) -> dict:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return {"type": "image", "mime": "image/jpeg",
            "data": base64.b64encode(buf).decode(), "caption": caption}


# ── Numeric core (pure — unit-tested by scripts/test_gravity_specialist.py) ──

def _fit_parabola(t: np.ndarray, y: np.ndarray) -> dict:
    """Least-squares y(t) = c0 + c1·t + c2·t². Returns fitted vertical accel
    a = 2·c2 (px/s², + = down), R², residuals and scale-free quality terms."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if len(t) < 3:
        return {"a": 0.0, "coef": np.zeros(3), "r2": 0.0, "rms": 0.0,
                "rms_frac": 0.0, "jump_frac": 0.0, "res": np.zeros_like(y)}
    t0 = t - t[0]                                   # conditioning
    coef = np.polyfit(t0, y, 2)
    res = y - np.polyval(coef, t0)
    ss_res = float((res ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else (1.0 if ss_res < 1e-6 else 0.0)
    vspan = max(float(y.max() - y.min()), 1.0)
    rms = float(np.sqrt((res ** 2).mean()))
    jump = float(np.abs(np.diff(res)).max()) if len(res) > 1 else 0.0
    return {"a": float(2.0 * coef[0]), "coef": coef, "r2": float(r2),
            "rms": rms, "rms_frac": rms / vspan, "jump_frac": jump / vspan,
            "res": res}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True → [(i0, i1)] inclusive."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[0], splits + 1])
    ends = np.concatenate([splits, [idx.size - 1]])
    return [(int(idx[s]), int(idx[e])) for s, e in zip(starts, ends)]


def _norm_track(tr: dict) -> dict:
    """Coerce a track's series to float arrays (in place-ish, returns tr)."""
    for k in ("x", "y", "vx", "vy", "speed"):
        tr[k] = np.asarray(tr[k], float)
    tr["frames"] = np.asarray(tr["frames"], int)
    tr["t"] = tr["frames"] / max(float(tr.get("_fps", 0)) or 1.0, 1e-6) \
        if "_fps" in tr else tr.get("t")
    return tr


def _check_segment(tr: dict, i0: int, i1: int, fps: float, H: float,
                   cfg: dict) -> tuple[dict, list[dict], list[dict]]:
    """Analyze one free-flight segment → (segment record, violations, floats)."""
    t = tr["t"][i0:i1 + 1]
    y = tr["y"][i0:i1 + 1]
    vy = tr["vy"][i0:i1 + 1]
    T = float(t[-1] - t[0])
    fit = _fit_parabola(t, y)
    a, r2 = fit["a"], fit["r2"]
    flat_px = max(4.0, 0.015 * H)
    # Mid-trajectory deviation a parabola of this curvature produces vs a
    # straight chord: |a|·T²/8. Below the flat threshold there is no
    # measurable gravity curvature at all.
    curv_px = abs(a) * T * T / 8.0

    seg = {"track_id": tr["id"], "label": tr["label"], "i0": i0, "i1": i1,
           "f0": int(tr["frames"][i0]), "f1": int(tr["frames"][i1]),
           "t0": round(float(t[0]), 3), "t1": round(float(t[-1]), 3),
           "a_fit": round(a, 1), "r2": round(r2, 4), "n": int(i1 - i0 + 1),
           "apex": False, "ballistic": False}
    violations: list[dict] = []
    floats: list[dict] = []

    def _v(vtype, score, detail, t_at, confirmed):
        return {"type": vtype, "track_id": tr["id"], "label": tr["label"],
                "t0": seg["t0"], "t1": seg["t1"], "t": round(float(t_at), 3),
                "frame": int(tr["frames"][i0 + int(np.argmin(np.abs(t - t_at)))]),
                "score": round(float(np.clip(score, 0.0, 1.0)), 3),
                "detail": detail, "confirmed": confirmed}

    t_mid = float(t[len(t) // 2])
    # Float suspicion grows with airborne duration — a long float is the real
    # levitation signature, and it wins the bounded VLM budget over blips.
    float_score = round(min(0.85, 0.45 + 0.15 * T / cfg["hover_min_s"]), 3)

    if curv_px < flat_px:
        # No gravity curvature: glide/hover over a meaningful duration.
        if T >= cfg["hover_min_s"]:
            floats.append(_v("float", float_score,
                             f"no measurable fall curvature over {T:.2f}s airborne "
                             f"(≈{curv_px:.1f}px deviation)", t_mid, None))
        return seg, violations, floats

    if a < 0:
        # Upward curvature. Only a CLEAR, well-fit upward acceleration is
        # anti-gravity; a weak drift (rising feather, thermal, camera creep)
        # is a VLM-gated float candidate at most.
        if curv_px >= 3.0 * flat_px and r2 >= 0.7:
            violations.append(_v("anti_gravity", 0.85,
                                 f"sustained UPWARD acceleration ({a:.0f} px/s², "
                                 f"R²={r2:.2f}) while airborne", t_mid, None))
        elif T >= cfg["hover_min_s"]:
            floats.append(_v("float", float_score,
                             f"drifting upward slowly ({a:.0f} px/s²) for "
                             f"{T:.2f}s airborne", t_mid, None))
        return seg, violations, floats

    # Downward-curving (gravity-like) segment — quality + consistency checks.
    seg["ballistic"] = a > 0 and r2 >= cfg["r2_min"]
    q = max((cfg["r2_min"] - r2) / 0.25,
            (fit["rms_frac"] - 0.06) / 0.10,
            (fit["jump_frac"] - 0.08) / 0.10)
    if q > 0.05:
        worst = i0 + int(np.argmax(np.abs(fit["res"])))
        violations.append(_v("non_parabolic_fall", q,
                             f"trajectory is not a clean parabola "
                             f"(R²={r2:.2f}, residual {fit['rms_frac']:.0%} of span, "
                             f"step {fit['jump_frac']:.0%})",
                             float(tr["t"][worst]), None))

    # Apex: vy crosses up→down (negative → positive in y-down coords).
    apex_k = next((k for k in range(3, len(vy) - 4)
                   if vy[k] < 0 <= vy[k + 1]), None)
    if apex_k is not None:
        seg["apex"] = True
        f_up = _fit_parabola(t[:apex_k + 1], y[:apex_k + 1])
        f_dn = _fit_parabola(t[apex_k + 1:], y[apex_k + 1:])
        # A leg only counts as evidence if its curvature is measurable over
        # its own duration (accel that bends the leg ≥ flat_px). Comparing
        # anything to an unmeasurably-flat leg is comparing noise.
        t_up = max(float(t[apex_k] - t[0]), 1e-6)
        t_dn = max(float(t[-1] - t[apex_k]), 1e-6)
        fl_up = 8.0 * flat_px / t_up ** 2
        fl_dn = 8.0 * flat_px / t_dn ** 2
        if f_up["a"] < -fl_up or f_dn["a"] < -fl_dn:
            violations.append(_v(
                "apex_asymmetry", 0.8,
                f"a leg of the arc curves UPWARD across the apex "
                f"({f_up['a']:.0f} px/s² up-leg, {f_dn['a']:.0f} px/s² down-leg)",
                float(t[apex_k]), None))
        elif f_up["a"] >= fl_up and f_dn["a"] >= fl_dn:
            ratio = max(f_up["a"], f_dn["a"]) / min(f_up["a"], f_dn["a"])
            if ratio > cfg["accel_ratio_max"]:
                violations.append(_v(
                    "apex_asymmetry",
                    0.4 + (ratio - cfg["accel_ratio_max"]) / (2 * cfg["accel_ratio_max"]),
                    f"gravity differs across the apex: {f_up['a']:.0f} px/s² up-leg "
                    f"vs {f_dn['a']:.0f} px/s² down-leg ({ratio:.1f}×)",
                    float(t[apex_k]), None))
        # Hang-time: a long |vy|≈0 plateau at the top (VLM-gated float).
        slow = np.abs(vy) < 0.15 * float(np.abs(vy).max())
        for s0, s1 in _runs(slow):
            if s0 <= apex_k <= s1 and float(t[s1] - t[s0]) >= 0.35:
                floats.append(_v("float", 0.6,
                                 f"{t[s1] - t[s0]:.2f}s hang-time at the apex",
                                 float(t[apex_k]), None))
                break
    elif len(t) >= 8:
        # No apex — compare fitted accel across the two halves.
        m = len(t) // 2
        f1_, f2_ = _fit_parabola(t[:m], y[:m]), _fit_parabola(t[m:], y[m:])
        if min(f1_["a"], f2_["a"]) <= 0 and max(f1_["r2"], f2_["r2"]) >= 0.5:
            violations.append(_v("inconsistent_accel", 0.8,
                                 f"acceleration flips sign mid-flight "
                                 f"({f1_['a']:.0f} → {f2_['a']:.0f} px/s²)",
                                 t_mid, None))
        elif min(f1_["a"], f2_["a"]) > 0:
            ratio = max(f1_["a"], f2_["a"]) / min(f1_["a"], f2_["a"])
            if ratio > cfg["accel_ratio_max"]:
                violations.append(_v(
                    "inconsistent_accel",
                    0.4 + (ratio - cfg["accel_ratio_max"]) / (2 * cfg["accel_ratio_max"]),
                    f"fall rate changes mid-air: {f1_['a']:.0f} → {f2_['a']:.0f} "
                    f"px/s² ({ratio:.1f}×)", t_mid, None))

    # Stage-2 corroboration: unexplained anomaly flags inside this window.
    flags = [i for i in list(tr.get("acc_flags") or []) + list(tr.get("rev_flags") or [])
             if i0 <= int(i) <= i1]
    if flags and violations:
        for v in violations:
            v["score"] = round(min(1.0, v["score"] + 0.1), 3)
            v["detail"] += f" · corroborated by {len(flags)} Stage-2 anomaly flag(s)"

    return seg, violations, floats


def _analyze_tracks(tracks: list[dict], contacts: list[dict], fps: float,
                    H: float, cfg: Optional[dict] = None) -> dict:
    """Pure numeric core: free-flight segmentation + all gravity checks.

    tracks: [{id, label, frames, x, y, vx, vy, speed, (t), (acc_flags),
              (rev_flags), (impacts)}] — image coords, y-down, height H.
    Returns {"segments", "violations", "float_candidates", "coherence"}.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    fps = max(float(fps), 1e-6)
    segments: list[dict] = []
    violations: list[dict] = []
    floats: list[dict] = []

    contact_frames: dict[int, set] = {}
    for ev in contacts or []:
        for key in ("track_a", "track_b"):
            contact_frames.setdefault(ev[key], set()).add(int(ev["frame"]))

    for tr in tracks:
        tr = _norm_track(tr)
        n = len(tr["frames"])
        if n < 3:
            continue
        if tr.get("t") is None:
            tr["t"] = tr["frames"] / fps
        tr.setdefault("label", f"obj {tr['id']}")

        grid = int(np.median(np.diff(tr["frames"]))) if n > 2 else 1
        gate = max(2 * max(grid, 1), int(round(0.08 * fps)))
        speed_floor = max(1.0, 0.15 * float(np.percentile(tr["speed"], 75)))

        # Frames near a tracked contact or an explained impact are not free
        # flight — they split the timeline (bounces stay out of the fits).
        blocked = set(contact_frames.get(tr["id"], set()))
        blocked |= {int(tr["frames"][int(i)]) for i in (tr.get("impacts") or [])
                    if 0 <= int(i) < n}
        near_block = np.zeros(n, bool)
        for bf in blocked:
            near_block |= np.abs(tr["frames"] - bf) <= gate

        off_floor = tr["y"] < cfg["floor_frac"] * H
        moving = tr["speed"] > speed_floor

        for i0, i1 in _runs(moving & off_floor & ~near_block):
            if (i1 - i0 + 1) < cfg["min_samples"]:
                continue
            if float(tr["t"][i1] - tr["t"][i0]) < cfg["min_airborne_s"]:
                continue
            seg, vs, fs = _check_segment(tr, i0, i1, fps, H, cfg)
            segments.append(seg)
            violations.extend(vs)
            floats.extend(fs)

        # Hover: parked mid-air (speed ≈ 0, off the floor) for a while.
        # Real levitation, or just resting on something — the VLM decides.
        # Named object tracks only: anonymous LK corner clusters are
        # background texture that "hovers" mid-frame constantly.
        if not tr.get("named"):
            continue
        for i0, i1 in _runs(~moving & off_floor):
            T_h = float(tr["t"][i1] - tr["t"][i0])
            if (i1 - i0 + 1) >= 4 and T_h >= cfg["hover_min_s"]:
                floats.append({
                    "type": "float", "track_id": tr["id"], "label": tr["label"],
                    "t0": round(float(tr["t"][i0]), 3),
                    "t1": round(float(tr["t"][i1]), 3),
                    "t": round(float(tr["t"][(i0 + i1) // 2]), 3),
                    "frame": int(tr["frames"][(i0 + i1) // 2]),
                    "score": round(min(0.85, 0.45 + 0.15 * T_h / cfg["hover_min_s"]), 3),
                    "confirmed": None,
                    "detail": f"hovering in place for {T_h:.2f}s off the ground"})

    # Galileo equivalence: simultaneously-airborne ballistic segments must
    # share the same fall acceleration regardless of mass.
    ballistic = [s for s in segments if s["ballistic"]]
    for sa, sb in combinations(ballistic, 2):
        if sa["track_id"] == sb["track_id"]:
            continue
        lo, hi = max(sa["t0"], sb["t0"]), min(sa["t1"], sb["t1"])
        if hi - lo < 0.25:
            continue
        ratio = max(sa["a_fit"], sb["a_fit"]) / max(min(sa["a_fit"], sb["a_fit"]), 1e-6)
        if ratio > cfg["equiv_tolerance"]:
            violations.append({
                "type": "unequal_fall", "pair": [sa["label"], sb["label"]],
                "track_id": sa["track_id"], "label": sa["label"],
                "t0": round(lo, 3), "t1": round(hi, 3),
                "t": round((lo + hi) / 2, 3),
                "frame": int(round((lo + hi) / 2 * fps)),
                "score": round(min(1.0, 0.3 + 0.7 * (ratio - cfg["equiv_tolerance"])
                                    / (2 * cfg["equiv_tolerance"])), 3),
                "detail": f"“{sa['label']}” falls at {sa['a_fit']:.0f} px/s² vs "
                          f"“{sb['label']}” at {sb['a_fit']:.0f} px/s² "
                          f"({ratio:.1f}× apart) in the same window",
                "confirmed": None})

    # Camera-motion coherence: many tracks sharing one velocity pattern means
    # global (camera) motion — pixel kinematics can't isolate the objects.
    coherence = 0.0
    if len(tracks) >= 3:
        by_frame = [{int(f): (float(vx), float(vy)) for f, vx, vy
                     in zip(tr["frames"], tr["vx"], tr["vy"])} for tr in tracks]
        corrs = []
        for da, db in combinations(by_frame, 2):
            common = sorted(set(da) & set(db))
            if len(common) < 6:
                continue
            va = np.array([da[f] for f in common]).T.ravel()
            vb = np.array([db[f] for f in common]).T.ravel()
            if va.std() < 1e-6 or vb.std() < 1e-6:
                continue
            c = float(np.corrcoef(va, vb)[0, 1])
            if np.isfinite(c):
                corrs.append(c)
        if corrs:
            coherence = float(np.median(corrs))

    return {"segments": segments, "violations": violations,
            "float_candidates": floats, "coherence": coherence}


# ── Rendering for VLM confirmation ────────────────────────────────────────────

def _labeled_tile(crop: np.ndarray, label: str, tile_h: int = 340) -> np.ndarray:
    s = tile_h / crop.shape[0]
    crop = cv2.resize(crop, (max(2, int(crop.shape[1] * s)), tile_h))
    band = np.full((30, crop.shape[1], 3), 20, np.uint8)
    cv2.putText(band, label, (5, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([band, crop], axis=0)


def _seg_strip(frames: list, tr: dict, i0: int, i1: int) -> Optional[np.ndarray]:
    """3 tiles across [i0, i1]: object circled, recent path drawn, timestamped."""
    xs, ys = tr["x"][i0:i1 + 1], tr["y"][i0:i1 + 1]
    Hf, Wf = frames[0].shape[:2]
    px = int(max(40, 0.25 * (xs.max() - xs.min() + 1)))
    py = int(max(40, 0.25 * (ys.max() - ys.min() + 1)))
    x0, x1 = max(0, int(xs.min()) - px), min(Wf, int(xs.max()) + px + 1)
    y0, y1 = max(0, int(ys.min()) - py), min(Hf, int(ys.max()) + py + 1)
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None
    tiles = []
    for si in (i0, (i0 + i1) // 2, i1):
        fi = int(tr["frames"][si])
        if fi >= len(frames):
            return None
        img = frames[fi].copy()
        pts = np.stack([tr["x"][i0:si + 1], tr["y"][i0:si + 1]], 1).astype(int)
        for k in range(1, len(pts)):
            cv2.line(img, tuple(pts[k - 1]), tuple(pts[k]), (0, 255, 255), 2,
                     cv2.LINE_AA)
        cur = (int(tr["x"][si]), int(tr["y"][si]))
        cv2.circle(img, cur, max(12, (y1 - y0) // 12), (60, 60, 230), 3, cv2.LINE_AA)
        tiles.append(_labeled_tile(img[y0:y1, x0:x1],
                                   f"t={tr['t'][si]:.2f}s"))
    gap = np.full((tiles[0].shape[0], 10, 3), 255, np.uint8)
    row = tiles[0]
    for tl in tiles[1:]:
        row = np.concatenate([row, gap, tl], axis=1)
    return row


def _video_height(path: str) -> float:
    """Cheap frame-height probe (no full decode). 0.0 if unknown."""
    try:
        cap = cv2.VideoCapture(path)
        h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if h > 0:
            return h
    except Exception:                                               # noqa: BLE001
        pass
    try:
        from PIL import Image
        with Image.open(path) as im:
            return float(im.size[1])
    except Exception:                                               # noqa: BLE001
        return 0.0


# ── Pipeline ──────────────────────────────────────────────────────────────────

async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg          = json.loads(settings) if settings else {}
    model        = str(cfg.get("model") or "createai:geminipro3_1")
    api_key      = str(cfg.get("api_key", "")).strip()
    max_checks   = max(1, int(cfg.get("max_checks", 3)))
    px_per_meter = float(cfg.get("px_per_meter", 0) or 0)
    g_tolerance  = float(cfg.get("g_tolerance", 0.20))
    core_cfg = {"min_airborne_s": max(0.1, float(cfg.get("min_airborne_s", 0.4))),
                "equiv_tolerance": max(1.05, float(cfg.get("equiv_tolerance", 1.5)))}

    auto_deps    = str(cfg.get("auto_deps", "agent")).lower()

    loop = asyncio.get_event_loop()
    vid = video_id(video_path)

    try:
        # ── 0. Evidence pre-step: plan + fetch missing Stage-2 producers ─────
        if auto_deps in ("agent", "rules"):
            from tools.evidence_planner import ensure_evidence
            async for ev in ensure_evidence(
                    video_path,
                    ["s2_object_tracker", "s2_trajectory_extractor"],
                    mode=auto_deps, model_key=model, api_key=api_key,
                    need_desc=(
                        "Gravity specialist: fits parabolas to free-flight "
                        "segments of moving objects. Needs per-object "
                        "trajectories with velocities/contacts/impacts "
                        "(trajectory_extractor), ideally named SAM3 tracks "
                        "(object_tracker) so hover checks work; a gravity "
                        "hypothesis time-window (hypothesis_generator) helps "
                        "focus its VLM budget.")):
                yield ev

        # ── 1. Kinematic tracks: evidence bus first, LK fallback ─────────────
        ev_traj = EVIDENCE.get(vid, "s2_trajectory_extractor")
        ev_tr   = EVIDENCE.get(vid, "s2_object_tracker")
        ev_hyp  = EVIDENCE.get(vid, "s2_hypothesis_generator")

        tracks: list[dict] = []
        contacts: list[dict] = []
        frames: Optional[list] = None
        sam3 = bool(ev_tr and ev_tr.get("mode") == "sam3")
        labels = {i: t.get("label") for i, t in enumerate((ev_tr or {}).get("tracks", []))}

        if ev_traj and ev_traj.get("trajectories"):
            fps = float(ev_traj.get("fps") or 30.0)
            contacts = ev_traj.get("contacts") or []
            for tj in ev_traj["trajectories"]:
                pos = np.asarray(tj["positions"], float)
                vel = np.asarray(tj["velocities"], float)
                tracks.append({
                    "id": int(tj["track_id"]),
                    "label": labels.get(int(tj["track_id"])) or f"obj {tj['track_id']}",
                    "named": bool(labels.get(int(tj["track_id"]))),
                    "frames": tj["frames"], "x": pos[:, 0], "y": pos[:, 1],
                    "vx": vel[:, 0], "vy": vel[:, 1], "speed": tj["speed"],
                    "acc_flags": tj.get("acc_flags"), "rev_flags": tj.get("rev_flags"),
                    "impacts": tj.get("impacts")})
            # Coordinate-frame height: SAM3 coords live in mask space.
            if sam3 and ev_tr.get("masks_png"):
                from tools.sam3 import decode_mask_png
                first = next(iter(next(iter(ev_tr["masks_png"].values())).values()))
                H = float(decode_mask_png(first).shape[0])
            else:
                H = _video_height(video_path)
            yield {"type": "log", "level": "info",
                   "text": f"Reusing {len(tracks)} kinematic track(s) + "
                           f"{len(contacts)} contact event(s) from the "
                           "Trajectory Extractor (evidence bus)."}
        else:
            yield {"type": "log", "level": "warn",
                   "text": "No trajectories on the evidence bus — falling back to "
                           "LK tracking (run Stage 2 first for contact-aware, "
                           "labeled evidence)."}
            from tools.tracking import get_tracks
            tr = await loop.run_in_executor(None, lambda: get_tracks(video_path))
            frames = tr["frames"]
            fps = float(tr["fps"])
            H = float(tr["meta"].get("H") or 0)
            min_disp = max(8.0, 0.02 * float(np.hypot(tr["meta"].get("W") or 0, H)))
            for ct in tr["tracks"]:
                x = np.asarray(ct["cx"], float)
                y = np.asarray(ct["cy"], float)
                if len(x) < 3 or float(np.hypot(x.max() - x.min(),
                                                y.max() - y.min())) < min_disp:
                    continue        # static background junk (same filter as Stage 2)
                k = np.ones(3) / 3.0
                x = np.convolve(np.pad(x, 1, mode="edge"), k, mode="valid")
                y = np.convolve(np.pad(y, 1, mode="edge"), k, mode="valid")
                t = np.asarray(ct["frames"], float) / max(fps, 1e-6)
                vx = np.gradient(x, t) if len(t) > 1 else np.zeros_like(x)
                vy = np.gradient(y, t) if len(t) > 1 else np.zeros_like(y)
                tracks.append({"id": int(ct["id"]), "label": f"obj {ct['id']}",
                               "named": False,        # LK corner clusters
                               "frames": ct["frames"], "x": x, "y": y,
                               "vx": vx, "vy": vy, "speed": np.hypot(vx, vy)})

        if not tracks:
            yield {"type": "log", "level": "warn",
                   "text": "No trackable moving objects — nothing to analyze."}
            yield {"type": "metric", "label": "Airborne segments", "value": "0",
                   "sub": "no tracks"}
            yield {"type": "severity", "label": "Gravity violation",
                   "value": 0, "color": "#4CAF50"}
            yield {"type": "done"}
            return
        if H <= 0:
            H = 1.15 * max(float(np.max(tr_["y"])) for tr_ in tracks)
            yield {"type": "log", "level": "warn",
                   "text": "Could not determine frame height — floor test uses "
                           "the deepest observed position instead."}

        # Stage 2's gravity hypothesis (if any) focuses the VLM budget.
        hyp_window = None
        if ev_hyp:
            h = next((h for h in ev_hyp.get("hypotheses", [])
                      if h.get("specialist") == "gravity"), None)
            if h:
                hyp_window = h.get("t_window")
                win = (f" t={hyp_window[0]}–{hyp_window[1]}s" if hyp_window else "")
                yield {"type": "log", "level": "info",
                       "text": f"Stage 2 hypothesis: gravity {h['confidence']:.0%}"
                               f"{win} — {h.get('reason', '')}"}

        # ── 2. Numeric core ───────────────────────────────────────────────────
        res = await loop.run_in_executor(
            None, lambda: _analyze_tracks(tracks, contacts, fps, H, core_cfg))
        segments, violations = res["segments"], res["violations"]
        floats, coherence = res["float_candidates"], res["coherence"]

        yield {"type": "log", "level": "info",
               "text": f"{len(segments)} free-flight segment(s) across "
                       f"{len(tracks)} track(s); {len(violations)} violation(s), "
                       f"{len(floats)} float candidate(s)."}
        for s in segments:
            yield {"type": "log", "level": "info",
                   "text": f"“{s['label']}” airborne t={s['t0']:.2f}–{s['t1']:.2f}s: "
                           f"a={s['a_fit']:.0f} px/s² (down), R²={s['r2']:.3f}"
                           + (" · apex" if s["apex"] else "")}
        for v in violations:
            yield {"type": "log", "level": "warn",
                   "text": f"{v['type'].upper()} — “{v['label']}” "
                           f"@ t={v['t']:.2f}s: {v['detail']}"}
        for fc in floats:
            yield {"type": "log", "level": "info",
                   "text": f"Float candidate — “{fc['label']}” @ t={fc['t']:.2f}s: "
                           f"{fc['detail']} (awaiting VLM confirmation)"}
        if coherence > 0.75:
            yield {"type": "log", "level": "warn",
                   "text": f"Tracks move coherently (r={coherence:.2f}) — likely "
                           "camera motion; pixel kinematics are unreliable, "
                           "severity capped at 50."}
        await asyncio.sleep(0)

        # ── 3. VLM confirms the ambiguous candidates (bounded) ────────────────
        from tools.vlm_router import ask_vision_json, key_status
        have_api, key_desc = key_status(model, api_key)
        queue = [v for v in violations if v["confirmed"] is None] + floats

        def _in_window(v):
            return bool(hyp_window and len(hyp_window) == 2
                        and hyp_window[0] - 0.5 <= v["t"] <= hyp_window[1] + 0.5)

        queue.sort(key=lambda v: (not _in_window(v), -v["score"]))
        queue = queue[:max_checks]

        if queue and have_api:
            if frames is None:
                if sam3 and ev_tr.get("num_frames"):
                    from tools.video import load_frames_rgb
                    rgb, _ = await loop.run_in_executor(
                        None, load_frames_rgb, video_path, int(ev_tr["num_frames"]))
                    frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in rgb]
                else:
                    from tools.video import load_frames
                    frames, _ = await loop.run_in_executor(None, load_frames, video_path)
            by_id = {t["id"]: _norm_track(dict(t)) for t in tracks}
            for t_ in by_id.values():
                t_.setdefault("t", np.asarray(t_["frames"], float) / max(fps, 1e-6))
            for v in queue:
                # unequal_fall carries a pair; the strip shows the first object
                # (its track_id) over the shared window — enough context to judge.
                tr_ = by_id.get(v["track_id"])
                if tr_ is None:
                    continue
                fsel = np.flatnonzero((tr_["t"] >= v["t0"]) & (tr_["t"] <= v["t1"]))
                if fsel.size < 2:
                    continue
                strip = _seg_strip(frames, tr_, int(fsel[0]), int(fsel[-1]))
                if strip is None:
                    continue
                try:
                    parsed = await ask_vision_json(
                        GRAVITY_PROMPT.format(label=v["label"], finding=v["detail"]),
                        strip, model, api_key)
                    support = str(parsed.get("support", "")).lower()
                    g_ok = parsed.get("gravity_ok")
                    conf = float(np.clip(float(parsed.get("confidence", 0.5)), 0, 1))
                    expl = str(parsed.get("explanation", ""))[:300]
                    thinking = str(parsed.get("reasoning", "")).strip()[:400]
                    obj_id = " · ".join(s for s in
                                        (str(parsed.get("object", "")).strip(),
                                         str(parsed.get("properties", "")).strip())
                                        if s)[:200]
                    if support in ("supported", "self_propelled") or g_ok is True:
                        v["confirmed"] = False
                        v["score"] = round(v["score"] * 0.25, 3)
                        reason = support if support in ("supported", "self_propelled") \
                            else "motion plausible for this object"
                        lvl, verdict = "info", f"plausible ({reason})"
                    elif support == "unsupported" and g_ok is False:
                        v["confirmed"] = True
                        v["score"] = round(max(v["score"], conf), 3)
                        lvl, verdict = "warn", "VLM CONFIRMS violation"
                    else:
                        lvl, verdict = "info", "unclear — score unchanged"
                    v["vlm_confidence"] = round(conf, 3)
                    v["explanation"] = expl
                    v["vlm_object"] = obj_id
                    v["vlm_reasoning"] = thinking
                    if obj_id:
                        yield {"type": "log", "level": "info",
                               "text": f"VLM identified: {obj_id}"}
                    if thinking:
                        yield {"type": "log", "level": "info",
                               "text": f"VLM reasoning: {thinking}"}
                    yield {"type": "log", "level": lvl,
                           "text": f"{verdict} — “{v['label']}” {v['type']} "
                                   f"@ t={v['t']:.2f}s ({conf:.0%}): {expl}"}
                    yield _jpeg_event(strip, f"“{v['label']}” {v['type']} "
                                             f"t={v['t0']:.2f}–{v['t1']:.2f}s — "
                                             f"{verdict} ({conf:.0%}) per {model}")
                except Exception as exc:                            # noqa: BLE001
                    yield {"type": "log", "level": "warn",
                           "text": f"VLM check failed @ t={v['t']:.2f}s: "
                                   f"{str(exc)[:140]} — keeping numeric score."}
                await asyncio.sleep(0)
        elif queue:
            yield {"type": "log", "level": "warn",
                   "text": f"No VLM credentials ({key_desc}) — {len(queue)} "
                           "candidate(s) reported unverified; float candidates "
                           "are dropped, not guessed."}

        # Floats only count once confirmed; unverified findings stay but are
        # capped below — drag/thrust/support could excuse any of them.
        violations.extend(f for f in floats if f["confirmed"] is True)
        unverified = bool(violations) and not any(v["confirmed"] is True
                                                  for v in violations)
        n_judged = sum(1 for v in queue if v.get("vlm_confidence") is not None)

        # ── 4. Optional absolute-g estimate (needs a user-supplied scale) ────
        ballistic = [s for s in segments if s["ballistic"]]
        if px_per_meter > 0 and ballistic:
            best = max(ballistic, key=lambda s: s["r2"])
            g_est = best["a_fit"] / px_per_meter
            dev = abs(g_est - 9.81) / 9.81
            yield {"type": "metric", "label": "g estimate",
                   "value": f"{g_est:.2f} m/s²",
                   "sub": f"“{best['label']}” · scale {px_per_meter:g} px/m — "
                          f"{dev:.0%} from 9.81 (estimate only)"}
            if dev > g_tolerance:
                yield {"type": "log", "level": "warn",
                       "text": f"Estimated g = {g_est:.2f} m/s² deviates {dev:.0%} "
                               f"from 9.81 (tolerance {g_tolerance:.0%}) — check "
                               "the px/m scale before trusting this."}

        # ── 5. Aggregate: signals, severity, plot, metrics ────────────────────
        signals = [{"frame": int(v["frame"]), "signal_type": f"gravity_{v['type']}",
                    "score": v["score"]} for v in violations]
        severity = int(round(100 * max([v["score"] for v in violations], default=0.0)))
        if unverified and severity > 70:
            severity = 70
            yield {"type": "log", "level": "warn",
                   "text": ("VLM examined the top candidate(s) and confirmed none — "
                            "severity capped at 70 from lower-ranked unexamined "
                            "findings; raise 'Max VLM confirmations' to examine more."
                            if n_judged else
                            "No violation is VLM-confirmed — severity capped at 70 "
                            "(drag/thrust/support could still explain these; add "
                            "VLM credentials for a firm verdict).")}
        if coherence > 0.75 and len(tracks) >= 3:
            severity = min(severity, 50)

        yield {"type": "signal", "source": "s3_gravity", "source_name": "Gravity",
               "fps": float(fps), "n_frames": int(max((t["frames"][-1] for t in tracks),
                                                      default=0)) + 1,
               "severity": severity,
               "type_severities": {
                   vt: int(round(100 * max([v["score"] for v in violations
                                            if v["type"] == vt], default=0.0)))
                   for vt in VIOLATION_TYPES},
               "signals": signals}

        if tracks:
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
                subplot_titles=("Vertical position (px, up = up) — shaded: airborne, "
                                "dashed: parabola fit",
                                "Vertical velocity (px/s, + = down)"))
            palette = ["#1a54c4", "#c05621", "#1a7a3c", "#7c3aed", "#be185d", "#0891b2"]
            for i, tr_ in enumerate(tracks):
                color = palette[i % len(palette)]
                t_ = np.asarray(tr_["frames"], float) / max(fps, 1e-6)
                fig.add_trace(go.Scatter(
                    x=t_.tolist(), y=(H - np.asarray(tr_["y"], float)).tolist(),
                    mode="lines", name=tr_["label"], legendgroup=tr_["label"],
                    line=dict(color=color, width=1.6)), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=t_.tolist(), y=np.asarray(tr_["vy"], float).tolist(),
                    mode="lines", name=tr_["label"], legendgroup=tr_["label"],
                    showlegend=False, line=dict(color=color, width=1.4)), row=2, col=1)
            for s in segments:
                fig.add_vrect(x0=s["t0"], x1=s["t1"], fillcolor="rgba(120,120,120,0.10)",
                              line_width=0)
                tr_ = next(t for t in tracks if t["id"] == s["track_id"])
                tt = np.asarray(tr_["frames"], float)[s["i0"]:s["i1"] + 1] / max(fps, 1e-6)
                yy = np.asarray(tr_["y"], float)[s["i0"]:s["i1"] + 1]
                cf = np.polyfit(tt - tt[0], yy, 2)
                fig.add_trace(go.Scatter(
                    x=tt.tolist(), y=(H - np.polyval(cf, tt - tt[0])).tolist(),
                    mode="lines", showlegend=False,
                    line=dict(color="#1a1917", width=1.2, dash="dash")), row=1, col=1)
            for v in violations:
                fig.add_vline(x=v["t"], line=dict(color="#E24B4A", dash="dot", width=1.2))
            fig.update_xaxes(title_text="Time (s)", row=2, col=1,
                             showgrid=True, gridcolor="#ebebeb")
            fig.update_yaxes(showgrid=True, gridcolor="#ebebeb", zeroline=False)
            fig.update_layout(
                title=dict(text="Gravity — Free-Flight Segments & Parabola Fits",
                           font=dict(size=15)),
                height=560, plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", y=1.09, x=0, font=dict(size=12)),
                margin=dict(l=65, r=40, t=100, b=50),
                font=dict(family="IBM Plex Sans, sans-serif", size=13))
            yield {"type": "plotly", "data": fig.to_json(),
                   "caption": "Shaded bands are analyzed free-flight windows; dashed "
                              "black curves are the ballistic fits; dotted red lines "
                              "mark gravity violations."}

        best_r2 = max((s["r2"] for s in segments), default=0.0)
        yield {"type": "metric", "label": "Airborne segments", "value": str(len(segments)),
               "sub": f"{len(tracks)} track(s) · min {core_cfg['min_airborne_s']:g}s"}
        yield {"type": "metric", "label": "Violations", "value": str(len(violations)),
               "sub": (", ".join(sorted({v['type'] for v in violations}))
                       if violations else "all airborne motion gravity-consistent")}
        yield {"type": "metric", "label": "Cleanest fit R²",
               "value": f"{best_r2:.3f}" if segments else "—",
               "sub": "1.0 = perfect parabola"}
        yield {"type": "metric", "label": "Float candidates", "value": str(len(floats)),
               "sub": f"{sum(1 for f in floats if f['confirmed'] is True)} confirmed "
                      f"by VLM · {sum(1 for f in floats if f['confirmed'] is False)} "
                      "explained as supported/self-propelled"}

        yield {"type": "severity", "label": "Gravity violation",
               "value": severity, "color": _sev_color(severity)}

        yield {"type": "result", "status": "ok", "severity": severity,
               "segments": segments, "violations": violations,
               "float_candidates": floats, "coherence": round(coherence, 3)}
        EVIDENCE.put(vid, "s3_gravity", {
            "severity": severity, "violations": violations,
            "segments": [{k: s[k] for k in ("label", "t0", "t1", "a_fit", "r2")}
                         for s in segments],
            "float_candidates": floats, "signals": signals,
            "coherence": round(coherence, 3)})

        if severity > 30:
            yield {"type": "log", "level": "warn",
                   "text": f"Gravity VIOLATION — severity {severity}%."}
        else:
            yield {"type": "log", "level": "success",
                   "text": "Airborne motion is consistent with constant downward "
                           "gravity."}
        yield {"type": "done"}

    except Exception as exc:                                        # noqa: BLE001
        import traceback
        yield {"type": "error", "text": f"{type(exc).__name__}: {exc}\n"
               f"{traceback.format_exc(limit=3)}"}
        yield {"type": "done"}

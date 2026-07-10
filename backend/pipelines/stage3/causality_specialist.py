"""
Stage 3 · Specialist — Causality / Temporal Drift Specialist
--------------------------------------------------------------
Confirms or rejects *causality* failures: effects that precede their causes,
globally time-reversed motion, and temporal-continuity artefacts (duplicated or
dropped frames). These are distinct from the per-object physics the other
specialists check — causality is about the ORDERING of events in time.

Signals (all derived from object kinematics, no GPU needed):

  1. Effect-before-cause — at a contact event, the struck object's motion change
     (acceleration spike) must occur AT or AFTER contact, never before. A strong
     acceleration response that leads the contact by more than the allowed lag is
     an effect preceding its cause.
  2. Global time-reversal — a natural velocity reversal is local to one object at
     a contact (a bounce). When a large fraction of objects reverse velocity on
     the SAME frame with no contact present, the segment is running backwards.
  3. Temporal drift — frame duplication shows up as a run of near-zero global
     motion sandwiched between motion (a stall); a dropped/spliced frame shows up
     as a synchronized position jump across most tracks on a single frame.

Evidence: reads `s2_trajectory_extractor` from the evidence bus when present
(richer kinematics); otherwise self-computes lightweight tracks via
tools.tracking.get_tracks so the specialist still runs standalone. Publishes its
verdict back to the bus as `s3_causality` for Stage 4.
"""
import asyncio, json
from typing import AsyncGenerator, Optional

import numpy as np
import plotly.graph_objects as go

from tools.evidence import EVIDENCE, video_id


# ── Normalise both evidence sources into one per-track structure ──────────────

def _robust_z(x: np.ndarray) -> np.ndarray:
    """Median/MAD z-score, NaN/zero-safe."""
    x = np.asarray(x, float)
    if x.size == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med)) or (np.std(x) or 1.0)
    return (x - med) / (1.4826 * mad + 1e-9)


def _tracks_from_evidence(ev: dict) -> list[dict]:
    """Map s2_trajectory_extractor trajectories → common structure."""
    out = []
    for tr in ev.get("trajectories", []):
        frames = [int(f) for f in tr["frames"]]
        out.append({
            "id": int(tr["track_id"]),
            "frames": frames,
            "pos":   np.asarray(tr.get("positions", []), float),
            "speed": np.asarray(tr.get("speed", []), float),
            "accel": np.asarray(tr.get("accel", []), float),
            "spike_idx": [int(i) for i in tr.get("acc_flags", [])],
            "rev_idx":   [int(i) for i in tr.get("rev_flags", [])],
        })
    return out


def _tracks_from_get_tracks(video_path: str, z_thresh: float) -> tuple[list[dict], float, int]:
    """Self-computed fallback: centroids → speed/accel → robust-z spikes."""
    from tools.tracking import get_tracks
    tr = get_tracks(video_path)
    fps = tr["fps"]
    n = tr["meta"]["n_frames"]
    out = []
    for ct in tr["tracks"]:
        f = [int(x) for x in ct["frames"]]
        pos = np.stack([np.asarray(ct["cx"], float), np.asarray(ct["cy"], float)], axis=1)
        if len(pos) < 3:
            continue
        vel = np.diff(pos, axis=0)                       # (k-1, 2)
        vmag = np.linalg.norm(vel, axis=1)
        speed = np.concatenate([[0.0], vmag]) * fps
        accel = np.concatenate([[0.0], np.diff(speed)])
        z = _robust_z(np.abs(accel))
        spikes = [int(i) for i in np.where(z > z_thresh)[0]]
        # Velocity reversal, jitter-guarded: a real reversal needs BOTH adjacent
        # velocity vectors to carry meaningful speed (a fraction of the track's
        # own median), so LK settling-jitter in the first frames isn't counted.
        vfloor = max(1.0, 0.4 * float(np.median(vmag[vmag > 1e-6])) if np.any(vmag > 1e-6) else 1.0)
        rev = []
        for i in range(1, len(vel)):
            if (np.dot(vel[i], vel[i - 1]) < 0
                    and vmag[i] > vfloor and vmag[i - 1] > vfloor):
                rev.append(i + 1)                        # +1: speed offset by one
        out.append({"id": int(ct["id"]), "frames": f, "pos": pos,
                    "speed": speed, "accel": accel,
                    "spike_idx": spikes, "rev_idx": rev})
    return out, fps, n


def _contacts_from_tracks(tracks: list[dict], pad: float = 24.0) -> list[dict]:
    """Fallback contact detection: rising edge of centroid proximity between pairs."""
    by_frame: dict[int, dict] = {}
    for t in tracks:
        for i, f in enumerate(t["frames"]):
            by_frame.setdefault(f, {})[t["id"]] = t["pos"][i]
    active: set = set()
    events = []
    for f in sorted(by_frame):
        present = by_frame[f]
        ids = sorted(present)
        touching = set()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                d = float(np.linalg.norm(present[ids[i]] - present[ids[j]]))
                if d <= pad:
                    pair = (ids[i], ids[j])
                    touching.add(pair)
                    if pair not in active:
                        events.append({"frame": f, "track_a": ids[i], "track_b": ids[j]})
        active = touching
    return events


# ── Detectors ─────────────────────────────────────────────────────────────────

def _frame_index(track: dict, frame: int) -> Optional[int]:
    try:
        return track["frames"].index(frame)
    except ValueError:
        return None


def _detect_effect_before_cause(tracks, contacts, fps, max_lag, window):
    """For each contact, check whether either participant's acceleration response
    LEADS the contact by more than max_lag frames (effect preceding cause)."""
    by_id = {t["id"]: t for t in tracks}
    violations = []
    for c in contacts:
        fc = int(c["frame"])
        for tid in (c["track_a"], c["track_b"]):
            t = by_id.get(tid)
            if not t or not t["spike_idx"]:
                continue
            spike_frames = [t["frames"][i] for i in t["spike_idx"] if i < len(t["frames"])]
            near = [sf for sf in spike_frames if abs(sf - fc) <= window]
            if not near:
                continue
            sf = min(near, key=lambda s: abs(s - fc))
            lag = sf - fc                                # negative → effect precedes cause
            if lag < -max_lag:
                idx = _frame_index(t, sf)
                violations.append({
                    "type": "effect_before_cause", "frame": int(sf), "contact_frame": fc,
                    "t": round(sf / max(fps, 1e-6), 3), "track": int(tid),
                    "lead_frames": int(-lag),
                    "detail": (f"track {tid} accelerates {-lag} frame(s) BEFORE its "
                               f"contact at frame {fc} (effect precedes cause)"),
                    "severity": float(min(1.0, 0.4 + (-lag - max_lag) / max(fps * 0.5, 1))),
                })
    return violations


def _detect_global_reversal(tracks, contacts, fps, frac_thresh, contact_guard):
    """Frames where a large fraction of active tracks reverse velocity at once
    with no nearby contact — a coherent time-reversal signature."""
    contact_frames = {int(c["frame"]) for c in contacts}
    rev_by_frame: dict[int, int] = {}
    active_by_frame: dict[int, int] = {}
    for t in tracks:
        for f in set(t["frames"]):
            active_by_frame[f] = active_by_frame.get(f, 0) + 1
        for i in t["rev_idx"]:
            if i < len(t["frames"]):
                f = t["frames"][i]
                rev_by_frame[f] = rev_by_frame.get(f, 0) + 1
    violations = []
    for f, nrev in sorted(rev_by_frame.items()):
        active = max(active_by_frame.get(f, 1), 1)
        frac = nrev / active
        near_contact = any(abs(f - cf) <= contact_guard for cf in contact_frames)
        if nrev >= 2 and frac >= frac_thresh and not near_contact:
            violations.append({
                "type": "global_time_reversal", "frame": int(f),
                "t": round(f / max(fps, 1e-6), 3),
                "n_reversed": int(nrev), "n_active": int(active),
                "detail": (f"{nrev}/{active} tracks reverse velocity simultaneously at "
                           f"frame {f} with no contact — segment appears time-reversed"),
                "severity": float(min(1.0, frac)),
            })
    return violations


def _global_motion_series(tracks, n_frames) -> np.ndarray:
    """Mean track speed per frame across the whole clip (0 where no track present)."""
    total = np.zeros(n_frames); count = np.zeros(n_frames)
    for t in tracks:
        for i, f in enumerate(t["frames"]):
            if 0 <= f < n_frames and i < len(t["speed"]):
                total[f] += t["speed"][i]; count[f] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), 0.0)


def _detect_temporal_drift(tracks, fps, n_frames, min_stall):
    """Frame duplication (a motion stall amid motion) and dropped/spliced frames
    (a synchronized jump across most tracks)."""
    drift = []
    motion = _global_motion_series(tracks, n_frames)
    moving = motion[motion > 1e-6]
    if moving.size >= 4:
        med = float(np.median(moving))
        stall_eps = 0.05 * med
        f = 1
        while f < n_frames - 1:                          # duplication: near-zero run amid motion
            if motion[f] <= stall_eps and motion[f] < med:
                start = f
                while f < n_frames and motion[f] <= stall_eps:
                    f += 1
                run = f - start
                before = motion[max(0, start - 2):start].max(initial=0.0)
                after  = motion[f:f + 2].max(initial=0.0)
                if run >= min_stall and before > med * 0.5 and after > med * 0.5:
                    drift.append({
                        "type": "frame_duplication", "frame_start": int(start),
                        "frame_end": int(f - 1), "t": round(start / max(fps, 1e-6), 3),
                        "n_frames": int(run),
                        "detail": (f"{run} frame(s) of near-zero motion (frames "
                                   f"{start}–{f-1}) between moving frames — likely duplicated"),
                        "severity": float(min(1.0, 0.3 + 0.1 * run)),
                    })
            else:
                f += 1

    disp_by_frame: dict[int, list] = {}                  # dropped frame: synchronized jump
    for t in tracks:
        pos = t["pos"]
        if len(pos) < 3:
            continue
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        z = _robust_z(step)
        for i in range(len(step)):
            if z[i] > 4.0 and step[i] > 1e-3:
                disp_by_frame.setdefault(int(t["frames"][i + 1]), []).append(int(t["id"]))
    n_tracks = max(len(tracks), 1)
    for f, ids in sorted(disp_by_frame.items()):
        if len(ids) >= 2 and len(ids) / n_tracks >= 0.6:
            drift.append({
                "type": "frame_drop", "frame_start": int(f), "frame_end": int(f),
                "t": round(f / max(fps, 1e-6), 3), "n_tracks": len(ids),
                "detail": (f"{len(ids)}/{n_tracks} tracks jump simultaneously at frame "
                           f"{f} — likely a dropped/spliced frame"),
                "severity": float(min(1.0, len(ids) / n_tracks)),
            })
    return drift


# ── Timeline plot ─────────────────────────────────────────────────────────────

def _timeline_fig(contacts, fps, effect_v, reversal_v, drift_v, duration):
    rows = ["Contacts", "Effect→cause", "Time reversal", "Temporal drift"]
    fig = go.Figure()

    def _scatter(events, row, color, symbol):
        if not events:
            return
        xs = [e.get("t", e.get("frame", 0) / max(fps, 1e-6)) for e in events]
        fig.add_trace(go.Scatter(
            x=xs, y=[row] * len(xs), mode="markers",
            marker=dict(size=13, color=color, symbol=symbol,
                        line=dict(width=1, color="white")),
            name=row, hovertemplate="%{y}<br>t = %{x:.2f}s<extra></extra>",
        ))

    _scatter(contacts, "Contacts", "#1a54c4", "circle")
    _scatter(effect_v, "Effect→cause", "#E24B4A", "triangle-left")
    _scatter(reversal_v, "Time reversal", "#d97706", "x")
    _scatter(drift_v, "Temporal drift", "#7c3aed", "square")

    fig.update_layout(
        title=dict(text="Causality Timeline — events & detected violations", font=dict(size=15)),
        height=330, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Time (s)", range=[-0.05 * max(duration, 1), duration * 1.05],
                   showgrid=True, gridcolor="#ebebeb", zeroline=False),
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(rows)),
                   showgrid=True, gridcolor="#f2f2f2"),
        margin=dict(l=110, r=40, t=60, b=50), showlegend=False,
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg           = json.loads(settings) if settings else {}
    max_lag       = max(1, int(cfg.get("max_causal_lag_frames", 3)))
    reversal_frac = float(cfg.get("reversal_fraction", 0.6))
    min_stall     = max(2, int(cfg.get("min_dup_run_frames", 3)))
    z_thresh      = float(cfg.get("z_threshold", 3.5))
    loop = asyncio.get_event_loop()

    yield {"type": "log", "level": "info", "text": "Causality Specialist — gathering kinematic evidence…"}
    await asyncio.sleep(0)

    vid = video_id(video_path)
    ev = EVIDENCE.get(vid, "s2_trajectory_extractor")
    if ev and ev.get("trajectories"):
        tracks   = _tracks_from_evidence(ev)
        fps      = float(ev.get("fps", 30.0))
        n_frames = int(ev.get("n_frames", 0)) or (max((max(t["frames"]) for t in tracks), default=-1) + 1)
        contacts = list(ev.get("contacts", []))
        yield {"type": "log", "level": "info",
               "text": f"Using Stage 2 trajectories: {len(tracks)} track(s), {len(contacts)} contact(s)."}
    else:
        yield {"type": "log", "level": "info",
               "text": "No Stage 2 evidence — self-computing tracks (run Trajectory Extractor first for richer input)."}
        await asyncio.sleep(0)
        try:
            tracks, fps, n_frames = await loop.run_in_executor(
                None, lambda: _tracks_from_get_tracks(video_path, z_thresh))
        except Exception as exc:  # noqa: BLE001
            yield {"type": "error", "text": f"Could not compute tracks: {exc}"}
            return
        contacts = _contacts_from_tracks(tracks)
        yield {"type": "log", "level": "info",
               "text": f"Self-computed {len(tracks)} track(s), {len(contacts)} contact(s)."}

    if len(tracks) == 0 or n_frames < 3:
        yield {"type": "log", "level": "warn",
               "text": "Not enough tracked motion to assess causality."}
        yield {"type": "metric", "label": "Causality verdict", "value": "inconclusive",
               "sub": "no trackable motion"}
        yield {"type": "severity", "label": "Causality violation severity", "value": 0, "color": "#9e9e9e"}
        EVIDENCE.put(vid, "s3_causality", {"verdict": "inconclusive", "severity": 0,
                                           "causality_violations": [], "temporal_drift": []})
        yield {"type": "done"}
        return

    duration = n_frames / max(fps, 1e-6)
    await asyncio.sleep(0)

    # ── Run the three detectors ───────────────────────────────────────────────
    window     = max(int(fps * 0.75), max_lag + 2)     # search radius around a contact
    effect_v   = _detect_effect_before_cause(tracks, contacts, fps, max_lag, window)
    reversal_v = _detect_global_reversal(tracks, contacts, fps, reversal_frac,
                                         contact_guard=max_lag + 1)
    drift_v    = _detect_temporal_drift(tracks, fps, n_frames, min_stall)

    for v in effect_v + reversal_v:
        yield {"type": "log", "level": "warn", "text": "⚠ " + v["detail"]}
    for d in drift_v:
        yield {"type": "log", "level": "warn", "text": "⏱ " + d["detail"]}
    if not (effect_v or reversal_v or drift_v):
        yield {"type": "log", "level": "success",
               "text": "No causality or temporal-drift violations detected."}
    await asyncio.sleep(0)

    # ── Timeline plot ─────────────────────────────────────────────────────────
    fig = _timeline_fig(contacts, fps, effect_v, reversal_v, drift_v, duration)
    yield {"type": "plotly", "data": fig.to_json(),
           "caption": ("Blue = contacts (causes). Red = an object's motion change that "
                       "precedes its contact. Orange = synchronized time-reversal. "
                       "Purple = duplicated/dropped frames.")}

    # ── Score & verdict — each channel contributes up to its own cap ──────────
    acausal_sev  = sum(v["severity"] for v in effect_v)
    reversal_sev = sum(v["severity"] for v in reversal_v)
    drift_sev    = sum(d["severity"] for d in drift_v)
    severity = int(min(100,
                       min(acausal_sev,  1.5) / 1.5 * 45 +
                       min(reversal_sev, 1.5) / 1.5 * 30 +
                       min(drift_sev,    1.5) / 1.5 * 25))
    n_viol = len(effect_v) + len(reversal_v) + len(drift_v)
    verdict = ("confirmed" if severity >= 50 else
               "suspected" if severity >= 20 else "rejected")
    color = "#E24B4A" if severity >= 50 else "#EF9F27" if severity >= 20 else "#4CAF50"

    yield {"type": "metric", "label": "Effect-before-cause", "value": str(len(effect_v)),
           "sub": "acceleration leads its contact"}
    yield {"type": "metric", "label": "Time-reversal events", "value": str(len(reversal_v)),
           "sub": "synchronized velocity flips"}
    yield {"type": "metric", "label": "Temporal-drift artefacts", "value": str(len(drift_v)),
           "sub": "duplicated / dropped frames"}
    yield {"type": "metric", "label": "Causality verdict", "value": verdict,
           "sub": f"{n_viol} violation(s), {len(contacts)} contact(s)"}
    yield {"type": "severity", "label": "Causality violation severity", "value": severity, "color": color}

    EVIDENCE.put(vid, "s3_causality", {
        "verdict": verdict, "severity": severity,
        "causality_violations": effect_v + reversal_v,
        "temporal_drift": drift_v,
        "n_contacts": len(contacts), "fps": fps, "n_frames": n_frames,
    })

    yield {"type": "log", "level": "success" if severity < 20 else "warn",
           "text": f"Causality verdict: {verdict} (severity {severity}/100, {n_viol} violation(s))."}
    yield {"type": "done"}

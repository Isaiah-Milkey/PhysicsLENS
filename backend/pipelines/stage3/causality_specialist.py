"""
Stage 3 · Specialist — Causality / Temporal Drift Specialist
--------------------------------------------------------------
A VLM *rule-checking agent*. Rather than asking one vague "is this suspicious?"
question, causality is decomposed into an explicit CHECKLIST OF RULES, each a
focused prompt the vision-language model checks and verifies against the frames:

    CHECK   — the rule is posed as a Yes/No question; we read P(Yes) straight
              from the model's next-token logits (a calibrated probability, not a
              self-written float).
    VERIFY  — if the rule fires, the model returns one sentence of concrete
              visual evidence (what it sees, which frame), so every flag is
              auditable.

Why rule-prompts instead of extracted numbers? Everything measurable from frames
is in PIXEL space — velocity/acceleration are relative, unscaled, and warped by
perspective and camera motion, so absolute-magnitude thresholds are meaningless.
Qualitative causality rules ("does an effect precede its cause?", "does the clip
play in reverse?") are scale-invariant and answerable by a VLM.

Corroboration: when Stage 2 (`s2_trajectory_extractor`) has run, its
scale-invariant GEOMETRIC signals — event ordering and synchronized velocity
reversal — cross-check the two rules that have a kinematic analogue, so geometry
verifies what the VLM claims. Publishes `s3_causality` to the bus for Stage 4.
"""
import asyncio, json
from typing import AsyncGenerator

import cv2
import numpy as np
import plotly.graph_objects as go

from tools.evidence import EVIDENCE, video_id
from tools.video    import load_frames, sample_frames


# ── The causality rule-checklist ──────────────────────────────────────────────
# Each rule: id, the physical principle it guards, the VLM check prompt, and the
# id of a Stage-2 geometric detector that can corroborate it (or None).
RULES = [
    {"id": "effect_before_cause", "law": "Cause must precede effect",
     "question": ("These frames are in time order. Does any object begin moving or "
                  "react BEFORE anything touches, hits, or pushes it — an effect "
                  "appearing before its cause?"),
     "geom": "effect_before_cause"},
    {"id": "time_reversal", "law": "The arrow of time",
     "question": ("Do any of these frames appear to play in REVERSE — scattered "
                  "pieces coming back together, spilled or splashed liquid returning "
                  "to its source, or a motion visibly undoing itself?"),
     "geom": "time_reversal"},
    {"id": "temporal_discontinuity", "law": "Temporal continuity",
     "question": ("Is there an abrupt break in the sequence — a frozen or repeated "
                  "moment, or a sudden jump — as if frames were duplicated or dropped?"),
     "geom": None},
    {"id": "cause_without_effect", "law": "Cause-effect coupling",
     "question": ("Does a clear physical cause occur — a collision, impact, or push — "
                  "WITHOUT its expected effect, so the affected object carries on as "
                  "if nothing happened?"),
     "geom": None},
    {"id": "spontaneous_change", "law": "No effect without a cause",
     "question": ("Does any object suddenly start moving, stop, or change direction "
                  "with NO visible cause acting on it?"),
     "geom": None},
]


# ── Geometric corroboration (scale-invariant, from Stage 2 evidence only) ─────

def _tracks_from_evidence(ev: dict) -> list[dict]:
    out = []
    for tr in ev.get("trajectories", []):
        out.append({
            "id": int(tr["track_id"]),
            "frames": [int(f) for f in tr["frames"]],
            "spike_idx": [int(i) for i in tr.get("acc_flags", [])],
            "rev_idx":   [int(i) for i in tr.get("rev_flags", [])],
        })
    return out


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    if x.size == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med)) or (np.std(x) or 1.0)
    return (x - med) / (1.4826 * mad + 1e-9)


def _tracks_from_get_tracks(video_path: str) -> tuple[list[dict], list[dict]]:
    """Self-computed motion signals when Stage 2 didn't run: centroids → accel
    spikes (robust-z) + jitter-guarded velocity reversals, plus proximity contacts.
    All signals are ordinal / relative-to-self, so they carry no pixel-scale units."""
    from tools.tracking import get_tracks
    tr = get_tracks(video_path)
    out = []
    for ct in tr["tracks"]:
        f = [int(x) for x in ct["frames"]]
        pos = np.stack([np.asarray(ct["cx"], float), np.asarray(ct["cy"], float)], axis=1)
        if len(pos) < 3:
            continue
        vel = np.diff(pos, axis=0)
        vmag = np.linalg.norm(vel, axis=1)
        accel = np.concatenate([[0.0], np.diff(np.concatenate([[0.0], vmag]))])
        spikes = [int(i) for i in np.where(_robust_z(np.abs(accel)) > 3.5)[0]]
        vfloor = max(1.0, 0.4 * (float(np.median(vmag[vmag > 1e-6])) if np.any(vmag > 1e-6) else 1.0))
        rev = [i + 1 for i in range(1, len(vel))
               if np.dot(vel[i], vel[i - 1]) < 0 and vmag[i] > vfloor and vmag[i - 1] > vfloor]
        out.append({"id": int(ct["id"]), "frames": f, "pos": pos,
                    "spike_idx": spikes, "rev_idx": rev})
    # proximity contacts (rising edge of centroid closeness)
    by_frame: dict = {}
    for t in out:
        for i, fr in enumerate(t["frames"]):
            by_frame.setdefault(fr, {})[t["id"]] = t["pos"][i]
    active, contacts = set(), []
    for fr in sorted(by_frame):
        present = by_frame[fr]; ids = sorted(present); touch = set()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if float(np.linalg.norm(present[ids[i]] - present[ids[j]])) <= 24.0:
                    pair = (ids[i], ids[j]); touch.add(pair)
                    if pair not in active:
                        contacts.append({"frame": fr, "track_a": ids[i], "track_b": ids[j]})
        active = touch
    return out, contacts


def _geom_effect_before_cause(tracks, contacts, max_lag=3, window=12) -> int:
    """Count contacts where a participant's acceleration spike LEADS the contact —
    a purely ordinal (scale-free) check of effect-before-cause."""
    by_id = {t["id"]: t for t in tracks}
    hits = 0
    for c in contacts:
        fc = int(c["frame"])
        for tid in (c.get("track_a"), c.get("track_b")):
            t = by_id.get(tid)
            if not t or not t["spike_idx"]:
                continue
            spike_frames = [t["frames"][i] for i in t["spike_idx"] if i < len(t["frames"])]
            near = [sf for sf in spike_frames if abs(sf - fc) <= window]
            if near and (min(near, key=lambda s: abs(s - fc)) - fc) < -max_lag:
                hits += 1
    return hits


def _geom_time_reversal(tracks, contacts, frac_thresh=0.6, guard=4) -> int:
    """Count frames where a large fraction of active tracks reverse velocity at
    once with no contact nearby — a scale-free time-reversal signature."""
    contact_frames = {int(c["frame"]) for c in contacts}
    rev_by, act_by = {}, {}
    for t in tracks:
        for f in set(t["frames"]):
            act_by[f] = act_by.get(f, 0) + 1
        for i in t["rev_idx"]:
            if i < len(t["frames"]):
                f = t["frames"][i]; rev_by[f] = rev_by.get(f, 0) + 1
    hits = 0
    for f, nrev in rev_by.items():
        frac = nrev / max(act_by.get(f, 1), 1)
        if nrev >= 2 and frac >= frac_thresh and not any(abs(f - cf) <= guard for cf in contact_frames):
            hits += 1
    return hits


def _geometry_signals(video_path: str, ev: dict) -> tuple[dict, str]:
    """Map rule id → geometric event count, from Stage 2 evidence if present else
    self-computed tracks. Returns (counts, source)."""
    if ev and ev.get("trajectories"):
        tracks, contacts, source = _tracks_from_evidence(ev), list(ev.get("contacts", [])), "stage2"
    else:
        try:
            tracks, contacts = _tracks_from_get_tracks(video_path)
            source = "self-computed"
        except Exception:  # noqa: BLE001
            return {}, "none"
    if not tracks:
        return {}, source
    return ({
        "effect_before_cause": _geom_effect_before_cause(tracks, contacts),
        "time_reversal":       _geom_time_reversal(tracks, contacts),
    }, source)


# ── Plot ──────────────────────────────────────────────────────────────────────

def _rules_fig(results, threshold):
    labels = [r["rule"]["law"] for r in results]
    scores = [r.get("score", r["p_yes"] or 0.0) for r in results]
    colors = ["#E24B4A" if r["fired"] else "#4CAF50" for r in results]
    fig = go.Figure(go.Bar(
        x=scores, y=labels, orientation="h", marker_color=colors,
        text=[f"{s:.2f}" for s in scores], textposition="outside",
    ))
    fig.add_vline(x=threshold, line=dict(color="#888", dash="dash", width=1.2),
                  annotation_text=f"fire ≥ {threshold:.2f}", annotation_position="top")
    fig.update_layout(
        title=dict(text="Causality Rules — P(violated) per rule", font=dict(size=15)),
        xaxis=dict(range=[0, 1.12], title="Model probability the rule is violated",
                   showgrid=True, gridcolor="#ebebeb"),
        height=90 + 52 * len(results), plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=30, t=55, b=40), showlegend=False,
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg          = json.loads(settings) if settings else {}
    model_key    = cfg.get("model", "qwen2.5-vl-7b")
    num_frames   = max(4, int(cfg.get("num_frames", 8)))
    threshold    = float(cfg.get("fire_threshold", 0.5))
    want_evidence = str(cfg.get("want_evidence", "true")).lower() not in ("false", "0", "no")
    loop = asyncio.get_event_loop()

    from tools.vlm_local import verify_rule, LOCAL_VLMS
    if model_key not in LOCAL_VLMS:
        model_key = "qwen2.5-vl-7b"
    info = LOCAL_VLMS[model_key]
    yield {"type": "log", "level": "info",
           "text": (f"Causality Specialist — VLM rule-checker on {info['label']} "
                    f"(~{info['vram_gb']} GB, no API key). {len(RULES)} rules.")}
    await asyncio.sleep(0)

    frames, fps = load_frames(video_path)
    if not frames:
        yield {"type": "error", "text": "Could not read any frames from the video."}
        return
    keyframes = sample_frames(frames, num_frames)
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in keyframes]

    # Scale-invariant geometric signals (Stage 2 evidence, else self-computed).
    # These are a CO-DETECTOR for the two motion rules a still-frame VLM can't see.
    vid = video_id(video_path)
    geom, geom_src = await loop.run_in_executor(
        None, lambda: _geometry_signals(video_path, EVIDENCE.get(vid, "s2_trajectory_extractor")))
    if geom:
        yield {"type": "log", "level": "info",
               "text": (f"Geometric motion signals ready ({geom_src}) — they can fire the "
                        "ordering & reversal rules the VLM cannot see in still frames.")}

    yield {"type": "log", "level": "info",
           "text": f"Checking {len(RULES)} causality rules over {len(keyframes)} frames "
                   "(first run loads the model — may take a minute)…"}
    await asyncio.sleep(0)

    # ── Run each rule: VLM CHECK (token-prob) + VERIFY (evidence) + geometry ──
    results = []
    for r in RULES:
        try:
            out = await loop.run_in_executor(
                None, lambda rr=r: verify_rule(rgb, rr["question"], model_key=model_key,
                                               n_sample=len(rgb), want_evidence=want_evidence))
        except Exception as exc:  # noqa: BLE001
            yield {"type": "log", "level": "warn", "text": f"Rule '{r['id']}' failed: {exc}"}
            out = {"p_yes": None, "evidence": ""}
        p = out["p_yes"]
        vlm_fired = p is not None and p >= threshold
        g = geom.get(r["geom"]) if r["geom"] else None
        geom_fired = bool(g and g > 0)
        fired = vlm_fired or geom_fired
        # effective score: VLM prob, lifted toward certainty when geometry agrees /
        # supplied by geometry when the VLM was blind to the motion.
        score = (p or 0.0)
        if geom_fired:
            score = max(score, 0.6 + min(0.35, 0.1 * g))
        results.append({"rule": r, "p_yes": p, "score": score, "fired": fired,
                        "vlm_fired": vlm_fired, "geom_fired": geom_fired,
                        "evidence": out["evidence"], "geom_support": g})

        p_txt = f"{p:.2f}" if p is not None else "n/a"
        if r["geom"] and geom:
            gtxt = f" · geometry: {g} event(s)"
        else:
            gtxt = ""
        if fired:
            src = "VLM+geometry" if (vlm_fired and geom_fired) else ("geometry" if geom_fired else "VLM")
            yield {"type": "log", "level": "warn",
                   "text": f"✗ {r['law']} — VIOLATED via {src} (P={p_txt}){gtxt}"
                           + (f" — {out['evidence']}" if out["evidence"] else "")}
        else:
            yield {"type": "log", "level": "info", "text": f"✓ {r['law']} — ok (P={p_txt}){gtxt}"}
        await asyncio.sleep(0)

    # ── Aggregate — severity from how many rules fire and how strongly ────────
    fired = [r for r in results if r["fired"]]
    # agreement (both VLM and geometry) nudges a fired rule's weight up
    weight = sum(min(1.0, r["score"] * (1.15 if (r["vlm_fired"] and r["geom_fired"]) else 1.0))
                 for r in fired)
    severity = int(min(100, 100 * weight / (0.8 * len(RULES))))
    if not fired:                                   # no rule fired → low residual from the max score
        mx = max((r["score"] or 0.0) for r in results)
        severity = int(min(100, 100 * mx * 0.4))
    verdict = ("confirmed" if severity >= 50 else
               "suspected" if severity >= 25 else "rejected")
    color = "#E24B4A" if severity >= 50 else "#EF9F27" if severity >= 25 else "#4CAF50"

    # ── Plot + evidence table ─────────────────────────────────────────────────
    yield {"type": "plotly", "data": _rules_fig(results, threshold).to_json(),
           "caption": ("Each bar is the model's probability that a causality rule is "
                       "violated, read from its Yes/No token logits. Red = fired.")}

    def _src(r):
        return ("VLM+geometry" if (r["vlm_fired"] and r["geom_fired"])
                else "geometry" if r["geom_fired"] else "VLM")

    def _ev(r):
        if r["evidence"]:
            return r["evidence"]
        if r["geom_fired"]:
            return f"{r['geom_support']} geometric event(s) — motion the VLM cannot see in stills"
        return "—"

    if fired:
        tbl = go.Figure(data=[go.Table(
            columnwidth=[24, 16, 60],
            header=dict(values=["<b>Rule violated</b>", "<b>Fired by</b>", "<b>Evidence</b>"],
                        fill_color="#7c3aed", font=dict(color="white", size=13),
                        align="left", height=30),
            cells=dict(values=[
                [r["rule"]["law"] for r in fired],
                [_src(r) for r in fired],
                [_ev(r) for r in fired]],
                fill_color=[["#f7f4fd", "#ffffff"] * len(fired)],
                align="left", font=dict(size=12.5), height=30)),
        ])
        tbl.update_layout(title=dict(text="Fired Causality Rules — evidence", font=dict(size=15)),
                          height=110 + 34 * len(fired), margin=dict(l=20, r=20, t=50, b=15),
                          paper_bgcolor="white", font=dict(family="IBM Plex Sans, sans-serif"))
        yield {"type": "plotly", "data": tbl.to_json(),
               "caption": "The rules that fired, whether the VLM or geometry caught it, and why."}

    top = max(fired, key=lambda r: r["score"]) if fired else None
    yield {"type": "metric", "label": "Rules fired", "value": f"{len(fired)}/{len(RULES)}",
           "sub": "causality rules violated"}
    yield {"type": "metric", "label": "Top violation", "value": top["rule"]["law"] if top else "—",
           "sub": _src(top) if top else "none fired"}
    n_agree = sum(1 for r in fired if r["vlm_fired"] and r["geom_fired"])
    yield {"type": "metric", "label": "VLM+geometry agree", "value": str(n_agree),
           "sub": "rules both methods fired"}
    yield {"type": "metric", "label": "Causality verdict", "value": verdict,
           "sub": f"severity {severity}/100"}
    yield {"type": "severity", "label": "Causality violation severity", "value": severity, "color": color}

    EVIDENCE.put(vid, "s3_causality", {
        "verdict": verdict, "severity": severity, "model": model_key,
        "rules": [{"id": r["rule"]["id"], "law": r["rule"]["law"],
                   "p_violated": r["p_yes"], "score": r["score"], "fired": r["fired"],
                   "fired_by": _src(r) if r["fired"] else None,
                   "evidence": r["evidence"], "geom_support": r["geom_support"]}
                  for r in results],
    })
    yield {"type": "log", "level": "success" if severity < 25 else "warn",
           "text": f"Causality verdict: {verdict} (severity {severity}/100, "
                   f"{len(fired)}/{len(RULES)} rules fired)."}
    yield {"type": "done"}

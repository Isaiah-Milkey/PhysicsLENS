"""
Non-VLM numeric motion signals on the VideoPhy-2 subset.

WHY: on the held-out split the VLM judges agree with EACH OTHER at rho=+0.43 but
with humans at only +0.18 — and three different prompts on the same model agree
at rho~0.85. They share a bias rather than making independent errors, which is
why both prompt evolution and 7-judge fusion failed to move the number. Breaking
that ceiling needs a channel that is not a VLM judgement at all.

These are the Stage-1/Stage-2 style signals PhysicsLENS already believes in,
computed here on the ORIGINAL videos (not the 8 frozen frames — 8 samples are
far too sparse for flow or acceleration):

  flow_mean/std      dense Farneback magnitude — how much motion there is
  flow_jerk          frame-to-frame change in mean flow; real motion is smooth,
                     generated motion often stutters or teleports
  flow_entropy       spatial dispersion of flow direction; incoherent fields
                     indicate parts of the scene moving independently of any
                     physical cause
  accel_p95          95th pct of tracked-keypoint acceleration; impulsive
                     accelerations without contact are the classic tell
  track_survival     fraction of LK keypoints surviving the clip — objects that
                     morph or vanish break tracking
  resid_energy       residual after fitting each track to constant velocity;
                     ballistic motion fits well, hallucinated motion does not

Written per clip so method_report.py can fuse them with the VLM signals.

Usage:
  python backend/scripts/motion_signals.py --workers 8
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))


def index_sources():
    """clip_id -> source video path, rebuilt the way videophy_prepare made ids."""
    from videophy_prepare import ensure_videos, index_videos
    idx = index_videos(ensure_videos())
    return idx


def _read(path, max_frames=48, max_side=320):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 1:
        cap.release()
        return []
    take = np.linspace(0, total - 1, min(max_frames, total)).astype(int)
    want = set(int(t) for t in take)
    out, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            h, w = fr.shape[:2]
            s = max_side / max(h, w)
            if s < 1:
                fr = cv2.resize(fr, (int(w * s), int(h * s)))
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        i += 1
    cap.release()
    return out


def signals(path):
    g = _read(path)
    if len(g) < 5:
        return None
    mags, ents = [], []
    for a, b in zip(g[:-1], g[1:]):
        f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(f[..., 0], f[..., 1])
        mags.append(float(mag.mean()))
        m = mag > max(mag.mean(), 1e-3)
        if m.sum() > 20:
            h, _ = np.histogram(ang[m], bins=16, range=(0, 2 * np.pi))
            p = h / max(h.sum(), 1)
            ents.append(float(-(p[p > 0] * np.log(p[p > 0])).sum()))
    mags = np.array(mags) if mags else np.zeros(1)

    # LK tracks for kinematics
    p0 = cv2.goodFeaturesToTrack(g[0], maxCorners=180, qualityLevel=0.01,
                                 minDistance=7)
    surv, acc, resid = 0.0, [], []
    if p0 is not None and len(p0) > 4:
        n0 = len(p0)
        traj = {i: [tuple(p0[i][0])] for i in range(n0)}
        alive = {i: True for i in range(n0)}
        cur, ids = p0, list(range(n0))
        for a, b in zip(g[:-1], g[1:]):
            if cur is None or len(cur) == 0:
                break
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, cur, None)
            if nxt is None:
                break
            keep = st.reshape(-1) == 1
            for j, k in enumerate(keep):
                if not k:
                    alive[ids[j]] = False
            cur = nxt[keep].reshape(-1, 1, 2)
            ids = [ids[j] for j, k in enumerate(keep) if k]
            for j, i in enumerate(ids):
                traj[i].append(tuple(cur[j][0]))
        surv = sum(alive.values()) / max(n0, 1)
        for i, pts in traj.items():
            if len(pts) < 5:
                continue
            P = np.array(pts, dtype=float)
            v = np.diff(P, axis=0)
            a2 = np.diff(v, axis=0)
            acc.append(float(np.linalg.norm(a2, axis=1).max()))
            # residual vs constant-velocity fit (ballistic motion fits well)
            t = np.arange(len(P))
            A = np.stack([t, np.ones_like(t)], 1).astype(float)
            for d in range(2):
                sol, *_ = np.linalg.lstsq(A, P[:, d], rcond=None)
                resid.append(float(np.mean((A @ sol - P[:, d]) ** 2)))
    return {
        "flow_mean": float(mags.mean()),
        "flow_std": float(mags.std()),
        "flow_jerk": float(np.abs(np.diff(mags)).mean()) if len(mags) > 1 else 0.0,
        "flow_entropy": float(np.mean(ents)) if ents else 0.0,
        "accel_p95": float(np.percentile(acc, 95)) if acc else 0.0,
        "track_survival": float(surv),
        "resid_energy": float(np.median(resid)) if resid else 0.0,
    }


def _job(t):
    cid, p = t
    try:
        return cid, signals(Path(p))
    except Exception as e:  # noqa: BLE001
        return cid, {"_error": str(e)[:80]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/videophy300")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    data = ROOT / a.data
    clips = json.loads((data / "manifest.json").read_text())["clips"]
    idx = index_sources()

    # clip_id was f"{model}__{stem}"[:110]; recover by matching that construction.
    jobs, miss = [], 0
    for c in clips:
        cid, gen = c["clip_id"], c["generator"]
        hit = None
        for stem, p in idx.items():
            if f"{gen}__{stem}"[:110] == cid:
                hit = p
                break
        if hit:
            jobs.append((cid, str(hit)))
        else:
            miss += 1
    print(f"matched {len(jobs)}/{len(clips)} source videos ({miss} missing)",
          flush=True)

    t0 = time.time()
    out = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, (cid, s) in enumerate(ex.map(_job, jobs), 1):
            out[cid] = s
            if k % 50 == 0:
                print(f"    {k}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

    ok = {k: v for k, v in out.items() if v and "_error" not in v}
    outp = data / "method_motion.json"
    outp.write_text(json.dumps({"method": "motion", "n": len(ok),
                                "signals": ok}, indent=1))
    print(f"\n  {len(ok)}/{len(jobs)} clips with signals  ({time.time()-t0:.0f}s)")

    # quick univariate readout vs human pc across ALL clips (not a held-out
    # claim — just which channels carry any signal at all)
    from vlm_rapidata_eval import spearman
    pcs = {c["clip_id"]: c["pc"] for c in clips}
    names = sorted(next(iter(ok.values())).keys())
    print(f"\n  univariate |rho| vs human pc (all {len(ok)}, orientation-free):")
    for nm in names:
        pv = [(v[nm], 6 - pcs[k]) for k, v in ok.items()]
        r = spearman([p for p, _ in pv], [q for _, q in pv])
        print(f"    {nm:16s} {r:+.3f}")
    print(f"\n  -> {outp}")


if __name__ == "__main__":
    main()

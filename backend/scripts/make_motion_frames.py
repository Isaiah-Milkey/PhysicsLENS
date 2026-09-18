"""
Render motion INTO the frames, so the specialist does not have to infer it.

WHY. Every diagnostic points the same way: the specialists are single-frame
appearance detectors. Shuffling the 8 frames changes almost nothing, on three
datasets and five judge models. Prompt-injecting motion statistics changes the
answer without improving it. Rank-fusing them helps two specialists. Learning a
combination over 33 features helps none. Those are all attempts to REPAIR a
judgement made from frames the model cannot temporally integrate.

This changes the input instead. If the model cannot integrate across frames,
draw the integration onto the frame it does look at.

Two representations, both computed from the ORIGINAL video on the same
wall-clock grid that stage_signals.py uses (real demos run 5-50 fps, so a fixed
frame stride would mean different things per clip):

  trails   Lucas-Kanade tracks drawn as fading tails on each sampled frame, plus
           a marker at the current point. A single frame then encodes where
           everything came FROM. Camera motion is subtracted first — otherwise a
           pan paints every tail identically and swamps object motion.

  diff     the frame, with a red overlay where it differs from ~0.4 s earlier.
           Cheap, no correspondence needed, and it directly exposes "something
           moved with nothing touching it", which is the single most common
           failure the annotators wrote down.

Output is a drop-in dataset: same manifest, same clip ids, frames replaced. Every
existing scorer runs on it unchanged.

Usage:
  python backend/scripts/make_motion_frames.py --data data/robotbench \
      --videos data/robotbench_videos --mode trails --out data/robotbench_trails
"""
import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from stage_signals import read_gray, SAMPLE_FPS, WINDOW_S  # noqa: E402

TRAIL = 6          # how many past positions to draw
LAG_S = 0.4        # difference lag in seconds


def read_color(path, max_side=512, fps=SAMPLE_FPS, window=WINDOW_S):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 24.0
    if total <= 1:
        cap.release()
        return []
    step = max(src / fps, 1.0)
    last = min(total - 1, int(window * src))
    want = set(int(round(x)) for x in np.arange(0, last + 1, step))
    out, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok or i > last:
            break
        if i in want:
            h, w = fr.shape[:2]
            s = max_side / max(h, w)
            if s < 1:
                fr = cv2.resize(fr, (int(w * s), int(h * s)))
            out.append(fr)
        i += 1
    cap.release()
    return out


def trails(color, gray):
    p0 = cv2.goodFeaturesToTrack(gray[0], maxCorners=140, qualityLevel=0.01,
                                 minDistance=7)
    if p0 is None or len(p0) < 4:
        return color
    T = len(gray)
    P = np.full((T, len(p0), 2), np.nan, np.float32)
    P[0] = p0[:, 0, :]
    cur, idx = p0, np.arange(len(p0))
    for t in range(1, T):
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(gray[t - 1], gray[t], cur, None)
        if nxt is None:
            break
        st = st.reshape(-1).astype(bool)
        P[t, idx[st]] = nxt[st][:, 0, :]
        cur, idx = nxt[st], idx[st]
        if len(cur) < 3:
            break
    sx = color[0].shape[1] / gray[0].shape[1]
    sy = color[0].shape[0] / gray[0].shape[0]
    # remove camera motion: without this a pan draws an identical tail on every
    # point and the object motion we care about is invisible inside it
    V = np.diff(P, axis=0)
    cam = np.nan_to_num(np.nanmedian(V, axis=1))
    shift = np.vstack([np.zeros((1, 2)), np.cumsum(cam, axis=0)])
    out = []
    for t in range(T):
        img = color[t].copy()
        for j in range(P.shape[1]):
            pts = []
            for k in range(max(0, t - TRAIL), t + 1):
                q = P[k, j]
                if np.isnan(q[0]):
                    continue
                q = q - shift[k] + shift[t]        # hold the camera still
                pts.append((int(q[0] * sx), int(q[1] * sy)))
            if len(pts) < 2:
                continue
            for m in range(1, len(pts)):
                f = m / len(pts)
                cv2.line(img, pts[m - 1], pts[m],
                         (0, int(80 + 175 * f), int(255 * (1 - f))), 1,
                         cv2.LINE_AA)
            cv2.circle(img, pts[-1], 2, (0, 255, 255), -1, cv2.LINE_AA)
        out.append(img)
    return out


def diffs(color, gray):
    lag = max(1, int(round(LAG_S * SAMPLE_FPS)))
    out = []
    for t in range(len(color)):
        img = color[t].copy()
        r = max(0, t - lag)
        d = cv2.absdiff(gray[t], gray[r])
        d = cv2.resize(d, (img.shape[1], img.shape[0]))
        d = cv2.GaussianBlur(d, (5, 5), 0)
        m = (d > max(12, float(np.percentile(d, 97)))).astype(np.uint8)
        ov = img.copy()
        ov[m > 0] = (0, 0, 255)
        out.append(cv2.addWeighted(ov, 0.45, img, 0.55, 0))
    return out


def one(job):
    cid, vpath, outdir, mode, n_out = job
    try:
        gray = read_gray(vpath)
        color = read_color(vpath)
        n = min(len(gray), len(color))
        if n < 6:
            return cid, 0
        gray, color = gray[:n], color[:n]
        frames = trails(color, gray) if mode == "trails" else diffs(color, gray)
        sel = np.linspace(0, len(frames) - 1, min(n_out, len(frames))).astype(int)
        d = Path(outdir)
        d.mkdir(parents=True, exist_ok=True)
        for k, i in enumerate(sel):
            cv2.imwrite(str(d / f"{k:03d}.jpg"), frames[int(i)],
                        [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        return cid, len(sel)
    except Exception as e:  # noqa: BLE001
        print(f"  fail {cid[:40]}: {str(e)[:60]}", file=sys.stderr)
        return cid, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="trails", choices=["trails", "diff"])
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()

    data = Path(a.data) if Path(a.data).is_absolute() else ROOT / a.data
    vdir = Path(a.videos) if Path(a.videos).is_absolute() else ROOT / a.videos
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    man = json.loads((data / "manifest.json").read_text())
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for c in man["clips"]:
        v = vdir / f"{c['clip_id']}.mp4"
        if v.exists():
            jobs.append((c["clip_id"], str(v),
                         str(out / "frames" / c["clip_id"]), a.mode, a.frames))
    print(f"{a.mode}: rendering {len(jobs)} clips -> {out}", flush=True)

    ok = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (cid, n) in enumerate(ex.map(one, jobs)):
            ok += n > 0
            if (i + 1) % 50 == 0:
                print(f"   {i+1}/{len(jobs)}", flush=True)

    # same manifest, so every existing scorer runs unchanged
    for c in man["clips"]:
        k = a.frames
        c["n_frames"] = k
        c["order_temporal"] = list(range(k))
        s = list(range(k))
        rng = np.random.default_rng(0)
        while k > 2 and s == list(range(k)):
            rng.shuffle(s)
        c["order_shuffled"] = s
    man["motion_mode"] = a.mode
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    for extra in ("stage_signals.json",):
        if (data / extra).exists():
            shutil.copy(data / extra, out / extra)
    print(f"\n{ok}/{len(jobs)} clips rendered -> {out}")


if __name__ == "__main__":
    main()

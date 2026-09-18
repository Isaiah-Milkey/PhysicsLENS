"""
Merge the robot annotation sets into one labels file the harness can stage.

Three video sets arrived together, generated from the SAME 82 real robot demos:
  cosmos3-nano/      81 AI clips, annotated (xlsx)
  Wan2.2_TI2V-5B/    81 AI clips, 64 annotated (csv)
  videos/            82 REAL demonstrations, no annotation needed

The real demos matter more than their lack of labels suggests. Every AI clip was
conditioned on a start frame from its real counterpart, so a real/AI pair shares
scene, objects, camera and task — everything except whether the physics is
genuine. That is the cleanest negative control this project has ever had: on
VideoPhy-2 every clip was AI, so "clean" never existed.

Real clips are staged with has_violation=0 and rating=4 (top of the 1-4 scale).
That IS an assumption — real footage can still look odd through a camera — so
they carry generator="real" and are kept out of the annotated tables unless a
test explicitly asks for them.

Clip ids are prefixed with the generator because both AI sets reuse the source
demo's filename, and an unprefixed id would silently collide and overwrite.

Usage:
  python backend/scripts/build_robotbench.py
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "robot_raw"
OUT = ROOT / "data" / "robotbench_labels.csv"

COLS = ["id", "source_file", "testset_id", "generator", "rating", "has_violation",
        "category", "rules", "caption", "task", "annotator", "action_completed"]


def prompt_for(d: Path, fn: str) -> tuple[str, str]:
    """Caption + task from the generator's per-clip sidecar json."""
    j = d / (Path(fn).stem + ".json")
    if not j.exists():
        return "", ""
    try:
        m = json.loads(j.read_text())
    except Exception:  # noqa: BLE001
        return "", ""
    return str(m.get("prompt") or "")[:1200], str(m.get("task") or "")[:300]


def norm(v) -> str:
    s = "" if v is None else str(v).strip()
    return "" if s.lower() in ("nan", "none") else s


def main():
    rows = []

    # ── cosmos: xlsx ────────────────────────────────────────────────────────
    c = pd.read_excel(ROOT / "cosmos_annotations1.xlsx")
    cdir = RAW / "cosmos3-nano"
    for _, r in c.iterrows():
        fn = norm(r.get("filename"))
        if not fn.endswith(".mp4") or not (cdir / fn).exists():
            continue
        cap, task = prompt_for(cdir, fn)
        issue = norm(r.get("description_of_issue"))
        rows.append(dict(
            id=f"cosmos__{Path(fn).stem}", source_file=fn,
            testset_id=norm(r.get("video_id")), generator="cosmos3-nano",
            rating=norm(r.get("physical_plausibility_1_4")),
            has_violation=1 if issue else 0,
            category=norm(r.get("physics_category")), rules=issue,
            caption=cap or norm(r.get("prompt")), task=task,
            annotator=norm(r.get("reviewer")),
            action_completed=norm(r.get("action_completed"))))

    # ── wan: csv ────────────────────────────────────────────────────────────
    w = pd.read_csv(ROOT / "wan2.2_annotations.csv")
    wdir = RAW / "Wan2.2_TI2V-5B"
    for _, r in w.iterrows():
        fn = norm(r.get("video_file"))
        if not fn.endswith(".mp4") or not (wdir / fn).exists():
            continue
        cap, task = prompt_for(wdir, fn)
        issue = norm(r.get("description_of_issue"))
        rows.append(dict(
            id=f"wan__{Path(fn).stem}", source_file=fn,
            testset_id=norm(r.get("testset_id")), generator="wan2.2-ti2v-5b",
            rating=norm(r.get("physical_plausibility_1_4")),
            has_violation=1 if issue else 0,
            category=norm(r.get("physics_category")), rules=issue,
            caption=cap, task=task, annotator=norm(r.get("annotator")),
            action_completed=norm(r.get("action_completed"))))

    # ── real demos: no annotation, physics is genuine by construction ───────
    idx = pd.read_csv(RAW / "meta" / "index.csv")
    for _, r in idx.iterrows():
        fn = norm(r.get("file"))
        if not fn or not (RAW / "videos" / fn).exists():
            continue
        rows.append(dict(
            id=f"real__{Path(fn).stem}", source_file=fn,
            testset_id=norm(r.get("id")), generator="real",
            rating="4", has_violation=0, category="", rules="",
            caption=norm(r.get("task")), task=norm(r.get("task")),
            annotator="", action_completed="yes"))

    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv(OUT, index=False)

    print(f"{len(df)} rows -> {OUT}\n")
    print(df.generator.value_counts().to_string())
    ann = df[df.generator != "real"]
    print(f"\nannotated AI clips: {len(ann)}")
    print("  ratings:", ann.rating.value_counts().sort_index().to_dict())
    print("  with a violation:", int((ann.has_violation == 1).sum()),
          "| clean:", int((ann.has_violation == 0).sum()))
    print("  missing rating:", int((ann.rating == "").sum()))
    print("  with caption:", int((ann.caption != "").sum()))

    cats = {}
    for v in ann.category:
        for p in re.split(r"[,;/]", v or ""):
            p = p.strip().lower()
            if p:
                cats[p] = cats.get(p, 0) + 1
    print("\n  category counts (both generators pooled):")
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"    {k:14s} {v}")

    # paired coverage: same source demo generated by both models
    both = set(df[df.generator == "cosmos3-nano"].testset_id) & \
        set(df[df.generator == "wan2.2-ti2v-5b"].testset_id)
    print(f"\n  source demos with BOTH generators annotated: {len(both)}")
    trip = both & set(df[df.generator == "real"].testset_id)
    print(f"  full triplets (real + cosmos + wan):           {len(trip)}")


if __name__ == "__main__":
    sys.exit(main())

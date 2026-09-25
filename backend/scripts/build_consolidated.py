"""
Ingest consolidated_annotations.csv (4 generators x observable/unobservable).

Resolves each annotation row to its video by (model, observability) -> directory,
not by filename alone: Wan and Cosmos reuse the source demo's filename, and the
updated drop suffixes Cosmos files with `_cosmos3-nano` while the CSV does not.

Joins the prompt tables from the prompt_generation branch so every clip carries
what it was SUPPOSED to show. For unobservable clips this is essential, not
decoration: the hidden property (e.g. "hydrophobic layer, repels liquid") is by
design not visible in the frames, so no judge can grade
`hidden_property_followed_1_4` without being told what the property is.

`pair_id` = "<testset_id>__<generator>" links an observable clip to its
unobservable twin — same source frame, same generator, only the hidden property
differs — which is what makes the paired obs/unobs analysis possible.

Output: data/consolidated_labels.csv + data/consolidated_videos/<clip_id>.mp4
symlinks, ready for eval_prepare.py.
"""
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "robot_raw2"
LINK = ROOT / "data" / "consolidated_videos"
OUT = ROOT / "data" / "consolidated_labels.csv"

GEN = {  # annotation model name -> (directory stem, short id)
    "Wan2.2": ("Wan2.2_TI2V-5B", "wan"),
    "cosmos3 nano": ("cosmos3-nano", "cosmos"),
    "Hunyuan 1.5": ("hunyuan15", "hunyuan"),
    "MAGI 4.5B distill": ("magi", "magi"),
}


def resolve(r):
    d, _ = GEN[r.model]
    d = RAW / (d + ("_unobs" if r.observability == "unobservable" else ""))
    f = d / r.video_file
    if f.exists():
        return f
    alt = d / (Path(r.video_file).stem + f"_{GEN[r.model][0]}.mp4")
    return alt if alt.exists() else None


def main():
    a = pd.read_csv(ROOT / "consolidated_annotations.csv")
    po = pd.read_csv(ROOT / "data/prompts/testset_prompts.csv").set_index("id")
    pu = pd.read_csv(ROOT / "data/prompts/testset_prompts_unobservable.csv").set_index("id")

    LINK.mkdir(parents=True, exist_ok=True)
    for p in LINK.iterdir():
        p.unlink()

    rows, miss = [], []
    for _, r in a.iterrows():
        src = resolve(r)
        if src is None:
            miss.append(f"{r.model}/{r.video_file}")
            continue
        g = GEN[r.model][1]
        unobs = r.observability == "unobservable"
        cid = f"{g}__{'unobs' if unobs else 'obs'}__{int(r.testset_id):03d}"
        (LINK / f"{cid}.mp4").symlink_to(src.resolve())
        P = pu if unobs else po
        pr = P.loc[r.testset_id] if r.testset_id in P.index else None
        get = (lambda k: "" if pr is None or pd.isna(pr.get(k)) else str(pr.get(k)))
        issue = "" if pd.isna(r.description_of_issue) else str(r.description_of_issue).strip()
        cat = "" if pd.isna(r.physics_category) else str(r.physics_category).strip()
        # a row can carry a category with a blank description (3 in the current
        # file); it is still a violation, so fall back to the category text
        # rather than silently dropping it from every per-specialist test
        if not issue and cat:
            issue = f"[category only] {cat}"
        rows.append(dict(
            id=cid, generator=g, observability=r.observability,
            testset_id=int(r.testset_id), pair_id=f"{int(r.testset_id):03d}__{g}",
            rating=r.physical_plausibility_1_4,
            hidden_followed=r.hidden_property_followed_1_4,
            action_completed=str(r.action_completed).strip().lower(),
            has_violation=1 if issue else 0, rules=issue,
            category="" if pd.isna(r.physics_category) else r.physics_category,
            annotator=r.annotator,
            caption=get("prompt"), task=get("task"), action=get("action"),
            hidden_property=get("hidden_property"),
            hidden_value=get("hidden_property_value"),
            expected_outcome=get("expected_outcome"),
            failure_signature=get("failure_signature")))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"{len(df)}/{len(a)} rows staged -> {OUT}  ({len(miss)} missing: {miss})")
    print(pd.crosstab(df.generator, df.observability))
    both = df.groupby("pair_id").observability.nunique()
    print(f"obs/unobs pairs: {(both == 2).sum()}")
    print(f"unobs with hidden_property text: "
          f"{(df[df.observability=='unobservable'].hidden_property != '').sum()}")
    print(f"captions present: {(df.caption != '').sum()}/{len(df)}")


if __name__ == "__main__":
    sys.exit(main())

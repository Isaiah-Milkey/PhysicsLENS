# PhysicsLENS Seed 100 — local demo corpus

**82 real robot demonstrations across 12 datasets**, screened by hand after download.

Started from the "PhysicsLENS Seed 100" manifest. Rows 99–100 (BEHAVIOR-1K simulation) were
excluded up front, giving 98 real demos; a manual screening pass then dropped 16 more.
`meta/seed100_manifest.json` still holds all 98 requested rows, each carrying a `kept` flag,
so the full original request stays on record and can be re-pulled.

## Layout

```
data/
  videos/           82 × .mp4  (H.264, one file per manifest row)   596 MB
  meta/
    seed100_manifest.json   all 98 requested rows + a `kept` flag per row
    screening.json          the 16 rows dropped in screening, with their source paths
    plan_<family>.json      resolved source paths, camera keys, provenance; each row
                            tagged KEPT or DROPPED-IN-SCREENING
    index.json / index.csv  the 82 surviving demos: file, camera, resolution, fps, frames
    verification.json       independent decode check of every file present
  logs/             per-family download logs (kept as the historical download record)
```

Filenames are `<id:03d>_<slug>.mp4`, so `007_*.mp4` is manifest row 7. Ids are **not**
contiguous — they are the original manifest row numbers, and the screening left gaps.

## What is here

| Demos | Dataset | | Demos | Dataset |
|---|---|---|---|---|
| 20 | Humanoid Everyday | | 4 | AgiBot World Alpha |
| 15 | RoboCOIN | | 3 | Mobile ALOHA |
| 12 | Unitree G1_Dex1 | | 2 | RH20T |
| 8 | Unitree UniBot-V1 | | 2 | Static ALOHA |
| 7 | NVIDIA GR00T-Teleop-GR1 | | 2 | Unitree Z1 dual-arm |
| 6 | Unitree G1_Dex3 | | 1 | DLR Sara Pour (OXE) |

Screening removed **DexWild, PH2D (UCSD) and RoboCook (OXE)** entirely — those datasets no
longer appear in the corpus.

## Coverage consequence of the screening

Worth knowing before using this as a specialist benchmark: **10 of the 16 dropped demos were
pour / faucet / spray tasks** — ids 1–6, 13, 20, 23, 24. Read against the source document's
per-specialist breakdown, that appears to remove *all six* Humanoid Everyday fluid-primary
episodes plus RoboCOIN, PH2D and DexWild fluid rows, so the **fluid specialist lost roughly
40% of its primary pool** and now leans on RoboCOIN pours, Unitree G1_Dex1 `Pour_Medicine`
and the UniBot-V1 syringe transfers. Deformation also lost both RoboCook Play-Doh episodes
and the air-pump compression (37, 38, 92).

This is an inference from the source document's aggregate specialist counts, not from
per-row specialist labels — the manifest never carried those. If fluid coverage matters,
the source document's own guidance was to *"swap a failed or retried demo for its neighbour
and keep the count"*, which is still possible: every dropped row's exact source path and
camera key is preserved in `meta/screening.json`.

## How it was pulled

Only the individual episode files listed in the manifest were fetched — never a whole repo.

- **LeRobot v2.x** — one mp4 per episode; downloaded directly.
- **LeRobot v3.0** — episodes are concatenated into large chunk files. Per-episode
  `from`/`to` timestamps were read from `meta/episodes/*.parquet`, the segment cut with
  ffmpeg (re-encoded for frame accuracy), and the large intermediate deleted. RH20T and the
  remaining Open-X row came this way, via community/official LeRobot mirrors — so no
  TensorFlow was needed for the RLDS-origin data.

One camera stream per row, matching the row's `view`: egocentric → head camera,
wrist/overhead → `cam_high`, third-person → global camera.

## Verification

All 82 files re-verified after screening: every file opens in cv2, its sequentially-decoded
frame count matches its container header, first/middle/last frames return non-empty arrays,
none is blank or static, and all 82 md5 hashes are distinct (no duplicate episodes).
Totals: 73,603 frames, 0.59 GB.

## Known deviations from the manifest

Verified against source data rather than assumed. Full text in `meta/plan_<family>.json`.
(Deviations that applied only to screened-out rows have been removed from this table; they
remain in `meta/screening.json` and the plan files.)

| Row(s) | Deviation |
|---|---|
| 73, 74 (RH20T) | 640×360, not 1280×720. 720p is RH20T's original capture resolution and is not distributed: the official release was downsampled by the authors to 320×180, and 640×360 (the LeRobot port) is the highest RH20T RGB obtainable without a >30 GB tar. Manifest said "any user"; `user_0005` was picked as the highest-rated run and kept across both scenes so they form a matched pair. |
| 68 (DLR Sara Pour) | 5 fps, not the documented 10 fps. The LeRobot mirror is internally consistent at 5 fps (info.json, stream rate and episode durations all agree). Decodes as 640×480. |
| 32–35, 57–59 (NVIDIA GR1) | Subset folder is `In-lab_Eval` (hyphen), not `Inlab_Eval`. Rows 34 and 35 had no episode index — resolved by task name in `EgoDex_Eval/meta/episodes.jsonl` to episodes 56 and 63, each the single episode with that task, as the manifest predicted. |
| 54 (RoboCOIN stack_blocks) | Valid file, but frames 329–611 (~46% of the clip) are a frozen post-task idle shot. Only the first ~11 s carry motion. |
| 82 (RoboCOIN clear_table_residue) | 960×720, unlike the other 640×480 RoboCOIN rows. Source-native, not a download error — relevant if a pipeline assumes fixed input size. |

`data/videos/` is gitignored; `meta/` and `logs/` are small and kept so any file — including
the screened-out ones — can be re-pulled from its recorded source path.

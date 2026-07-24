# EWMBench ground-truth robot videos (real world)

Real robot-manipulation footage from the **EWMBench** benchmark
(`agibot-world/EWMBench` on HuggingFace, `gt_dataset.tar`), sampled from the
AgiBot World dataset. First-person robot view, 640×480. License: CC-BY-NC-SA-4.0.

Source is a JPG frame sequence per episode; re-encoded here to mp4 with
`ffmpeg -framerate 30` (AgiBot World records RGB at 30 fps — EWMBench itself
does not restate the rate). Filenames are `<task-slug>-<episode_id>.mp4`.

| Task | Prompt | Episodes |
|------|--------|----------|
| 367 | Place the toast on the plate. | 649524, 649559, 650191 |
| 392 | Grab the cup brush from the storage box on the counter with the right arm. | 651464, 664600, 681186 |
| 497 | Push open the freezer door with the right arm. | 766602, 773025 |
| 511 | Pass the held showerhead to right arm with left arm. | 743247, 743964, 744776 |
| 543 | Place the held item into the box with left arm. | 798615, 798749, 807480 |
| 558 | Place the golden kettle / transparent teapot held in the left arm back on the table. | 787136, 789120, 791059 |
| 574 | Pick up the golden/blue-hat kettle with right or left arm. | 808158, 824748, 834014 |

Note: episode `497/773496` is missing — the upstream `gt_dataset.tar` on
HuggingFace is truncated (unexpected EOF) and only 67 of its 273 frames
survive, in both the official repo and the `Physis-AI/wm-eval-gt-ewmbench`
repack. 20 of the 21 GT episodes are complete and included here.

The matching AI-generated side (same tasks, world-model output) is in the same
HF repo as `generated_samples.tar` (~139 MB) if we ever want real/AI pairs.

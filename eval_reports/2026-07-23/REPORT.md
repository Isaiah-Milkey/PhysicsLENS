# PhysicsLENS framework evaluation — real-video false-positive analysis

**Date:** 2026-07-24 · **Set:** all 30 videos under `test_videos/real/`
(20 EWMBench robot-manipulation episodes, 6 Physics-IQ lab clips, 4 Wikimedia
clips) · **Pipelines:** all 17, full stage order per video (bus-connected,
one process) · **Harness:** `backend/scripts/eval_framework.py`

Config = shipped defaults except: all VLM calls routed to **Gemini 3.1 Pro via
CreateAI** (user request; shipped defaults use local Qwen for screening/naming
and Gemini Flash for specialists), overlay-video rendering off, gravity's
evidence planner off (deps ran explicitly). `s3_causality` ran as a separate
local-Qwen pass (CreateAI strips logprobs — probed 2026-07-23, all providers,
both endpoints — so its logit mechanism cannot go through the API).

Side data: 3 AI clips (lumiere bee_honey / beer_pouring / big_wave) evaluated
with the identical config before the run was scoped to real-only. n=3 is
anecdotal — used for contrast, not conclusions.

---

## Headline

**The framework cannot currently tell real from AI video. The final
diagnostic score is flat — and slightly inverted.** Every real video landed
at 84–92 "physics consistency"; AI clip `bee_honey` scored **93.75, more
consistent than all 30 real videos**. The cause is not one bad detector but
uniform false positives from three pipelines (temporal, fluid, deformation)
that fire at ~80–100 on *everything*, drowning the discriminative signal in
the aggregate.

Mean max-severity on the 30 real videos (0 = correctly quiet):

| Pipeline | Real mean | Verdict on real footage |
|---|---|---|
| s1_temporal | **94.9** | Broken — 100 on 27/30; fixed px/s² threshold |
| s3_fluid | **96.4** | Broken — fabricates a "fluid region" when no fluid exists |
| s3_deformation | **79.2** | Broken — inherits tracker mask drift as "morphing" |
| s1_vlm (Gemini 3.1 Pro) | **72.2** | Heavy FP — "morphing/interpenetration" on real robot clips |
| s3_momentum | **44.0** | Scope gap — actuated (robot) contacts read as momentum-from-nowhere |
| s3_gravity | 23.3 | Policy flaw — VLM rejects candidates, score stays 70 anyway |
| s3_friction | 12.7 | Track jitter confirmed by identity-based VLM reasoning |
| s2_object_tracker | 37.4 | Moderate drift scores (feeds deformation's FPs) |
| s3_collision | **6.2** | Healthy — quiet on real, one containment miss |
| **s3_causality** | **5.4** | **Best specialist — zero scores ≥50 on real videos** |
| s1_optical_flow / embeddings / camera_motion | 9.1 / 12.2 / 5.3 | Healthy, quiet |
| s2_trajectory / localizer | 10.8 / 0.0 | Healthy, quiet |

---

## Root causes (dissected from the saved streams)

**1. `s1_temporal` — absolute threshold in pixel units.**
`grab-cup-brush-651464`: peak acceleration 37,526 px/s² vs a fixed default
threshold; 156/209 frames "anomalous" → severity 100. Any real motion at
640×480/30fps trivially exceeds a fixed px/s² number. Telling: its only low
scores in the whole eval (21–25) were the two *synthetic animation* webms —
it currently detects motion roughness, i.e. it's an **inverted** detector.

**2. `s3_fluid` — fabricates evidence instead of abstaining.**
`pick-up-kettle-808158` (no liquid anywhere): "No fluid-like subject named by
the Object Tracker — falling back to a motion-energy region", then advects
379 points through the *robot arm's* motion field and flags 13+ "droplet
paths defying gravity" (actuated arm points don't follow parabolas, by
design). Fired 95–100 on 29/30 real videos. The no-fluid fallback should
abstain, not invent a fluid.

**3. `s3_deformation` — blind to tracking quality.**
`grab-cup-brush-651464`: "robotic arm #1 → changed (99%): transforms from a
white robotic arm into a dark…" — the SAM3 mask slid off the arm onto another
object; DINOv2 correctly reports the crop changed; the VLM correctly reports
the crop changed. Both are right about the crop and wrong about the world:
nothing deformed, the *track* failed. No track-quality gate exists.

**4. `s1_vlm` — Gemini 3.1 Pro over-triggers on tiled robot frames.**
`pick-up-kettle-808158`: "gripper suddenly morphs", "objects passing through
each other", confidence 98% → severity 95, on real footage. 8 keyframes from
157 frames = large temporal jumps presented as one composite; occlusions and
sampling gaps read as morphing/interpenetration. Real mean 72.2 vs AI mean
87.5 (n=3) — no usable separation.

**5. `s3_momentum` — no concept of actuated motion.**
`place-item-in-box-798615` (sev 99): transfer anomalies flagged at every
arm–object contact ("momentum appears from nowhere"). A motorized gripper
*injects* momentum continuously; conservation between passive bodies is the
wrong model for robot video — every EWMBench manipulation clip is a
guaranteed false positive. Also flags contacts between *static adjacent*
objects (mask-noise phantom transfers).

**6. `s3_gravity` — the VLM gate works, the scoring policy overrides it.**
`pass-showerhead-743247`: candidates were track-noise junk ("marble wall #2:
acceleration flips sign mid-flight" — walls don't fall). The VLM examined the
top candidates and **confirmed none** — but severity stayed at 70 via the
"unexamined lower-ranked findings" cap. 9 of 30 real videos got exactly 70
this way. When candidate generation is noisy, cap-70-if-unexamined converts
noise into a permanent false positive.

**7. `s3_friction` — the VLM confirms hypotheticals, not observations.**
`place-teapot-on-table-789120` (sev 90): a *potted plant* "sped up from
~246 px/s" (track jitter/occlusion); VLM confirms: "the potted plant is an
inanimate object and there is no visible external force". The VLM validated
"IF the measurement is true, it's a violation" — it never checked whether the
plant actually moved. The confirm prompt presents the numeric claim as fact.

**8. `s3_collision` — containment read as interpenetration (1 case).**
`ball-in-basket` (sev 95): ball drops *into* a crate → 67% mask overlap →
"volume unnaturally merging". A container's mask legitimately overlaps its
contents in 2D. Otherwise collision was the quietest mask-based specialist
(0 on 27/30) and the best separator in the side data (AUC 0.78, n=3).

**9. `bouncing_ball.webm` — degenerate-content crashes.** Flat synthetic
ball: optical-flow found no keypoints, SAM3 found no subjects → 3 pipelines
errored instead of degrading. (This file is an animation mislabeled into
`real/`, as is `bouncing_ball_anim.webm` — their scores were excluded from
the false-positive counts above where relevant.)

---

## Stage-wise failure breakdown (quantified)

Counts are over the 30 real videos; "avg score" is the mean max-severity of
the videos exhibiting that reason (0 would be correct on real footage).

### Stage 1 — Screening

| Failure reason | Where | Count | Avg score |
|---|---|---|---|
| Numeric threshold in absolute px/s² — any real motion exceeds it | `s1_temporal` | 28/30 videos | 100 |
| Prompt frames the task as AI-detection: *"deciding whether a short video is REAL footage or AI-GENERATED (AI clips often break physics)"* — primes suspicion on unusual-looking (robot POV) footage | `s1_vlm` | 23/30 videos claim violations | 91 |

Wrong-reasoning clusters inside the `s1_vlm` verdicts (each a claimed
violation on real footage, all artifacts of sparse 8-frame tiling +
occlusion):

| Claimed violation | Count |
|---|---|
| "object morphing" / "rigid body morphing" / "rigid body deformation" | 10 |
| "objects passing through each other" | 3 |
| "object permanence" | 3 |
| "vanishing object" | 1 |

### Stage 2 — Localization

| Failure reason | Where | Count | Avg score |
|---|---|---|---|
| SAM3 mask drift (mask slides to another object mid-track) | `s2_object_tracker` | drift ≥50 on 7/30 | 37 (all) |
| Scene-inconsistent groundings become "objects" (e.g. *"marble wall #2"* later analyzed as a falling body by gravity) | naming → all of stage 3 | seen in sampled streams (not auto-countable) | — |

Stage 2's own severities are honest (trajectory 10.8, localizer 0.0); its
failure mode is *silent evidence corruption* — drifted masks and phantom
subjects are handed to stage 3 with no quality flag, which is where they
become physics violations.

### Stage 3 — Specialists

| Failure reason | Where | Count | Avg score |
|---|---|---|---|
| Never abstains: *"No fluid-like subject named by the Object Tracker — falling back to a motion-energy region"* fired on **every video**, then actuated/non-ballistic point paths are read as droplets (877 `ballistic_arc` flags total) | `s3_fluid` | 30/30 videos | 96 |
| Actuated contacts treated as passive collisions ("momentum appears from nowhere" at arm–object contacts) | `s3_momentum` | 38 flags | 93 (13 videos ≥70) |
| Phantom transfers between static adjacent objects (mask noise at "contacts") | `s3_momentum` | 33 flags | (same videos) |
| Prompt orders a difference report — *"your job is to verify it and say what differs"* — with **no tracking-artifact escape option**; VLM answered "changed" 101 of 125 times | `s3_deformation` | 28/30 videos | 79 |
| VLM physically cannot verify the claim from the evidence shown: speed changes judged from **two still frames** (*"LEFT is BEFORE, RIGHT is AFTER … An automated motion metric flagged this: {desc}. Verify it."*) → falls back to object-identity reasoning ("a potted plant can't self-accelerate") | `s3_friction` | 12 flags, 3 VLM-confirmed | 90 (confirmed) |
| Score-floor policy contradicts the VLM verdict: *"No violation is VLM-confirmed — severity capped at 70"* (7 videos) and *"confirmed none — capped at 70 from lower-ranked unexamined findings"* (2 videos) | `s3_gravity` | 9/30 videos at exactly 70 | 70 |
| Junk gravity candidates from track noise feeding the above: NON_PARABOLIC ×13, UNEQUAL_FALL ×8, INCONSISTENT_ACCEL ×7, ANTI_GRAVITY ×7, APEX ×2 (includes non-falling subjects like walls) | `s3_gravity` | 37 candidates | — |
| 2-D containment read as interpenetration (ball *inside* crate = 67% mask overlap → "volume unnaturally merging") | `s3_collision` | 1/30 videos | 95 |

Contrast: `s3_causality` fired individual rules 31 times across the set but
its calibrated probability scoring kept every video ≤20 (mean 5.4) — the only
specialist whose *scoring* survives noisy inputs.

### Stage 4 — Aggregation

| Failure reason | Where | Evidence |
|---|---|---|
| Uniform false positives dominate the average: every real video lands at 84–92 "consistency"; AI clip scored 93.75 | `s4_report` | flat, slightly inverted score |
| Mixed signals in one report: overall max-severity events read 100 on real videos while the consistency metric says ~87 | `s4_report` | severity event vs metric disagree |
| No test-reliability weighting: a test that fires ≥90 on >80% of real reference footage (temporal, fluid) carries the same weight as one that is quiet (causality, collision) | `s4_report` | by construction |

### Reading of the pattern

Three distinct failure *mechanisms*, in order of damage:

1. **Wrong evidence, honest judges** (deformation, friction, momentum,
   gravity candidates): stage 2 hands stage 3 corrupted tracks with no
   quality signal; every downstream judgment is then correct about the crop
   and wrong about the world.
2. **Wrong prompt contracts** (`s1_vlm` real-vs-AI framing; deformation's
   "say what differs"; friction's "verify [the measurement]" against stills
   that can't show it): the VLM is never given "the measurement may be
   false" as an answer.
3. **Wrong scoring policy** (gravity's 70-floor after rejection; fluid's
   refuse-to-abstain fallback; report's unweighted averaging): even where
   detection or verification behaves, the score ignores it.

---

## What actually works

- **`s3_causality` (local Qwen, Yes/No logits): mean 5.4, zero ≥50 on real
  videos** — the checklist-plus-logits design is the most trustworthy
  specialist in the framework, and the cheapest per video (~7 s).
- **Classical stage-1 signals** (optical flow, embeddings, camera motion) are
  honest: quiet on real, and in the side data embeddings/optical-flow rose on
  AI clips (up to 100 on `big_wave`).
- **Collision's numeric core** — near-silent on real footage.
- **Stage-2 plumbing** (tracker→trajectory→localizer→bus) ran 480 pipeline
  executions with only 3 content-degenerate errors; evidence propagation
  worked every time.

## Recommended fixes, in impact order

1. **Fluid: abstain when no fluid-like subject is named** (delete the
   motion-energy fallback or make it report-only). Removes ~29/30 FPs.
2. **Temporal: normalize acceleration by object/frame scale** (relative to
   median motion, like the robust-z used elsewhere). Removes ~27/30 FPs.
3. **Gravity: drop the unexamined-findings 70-cap when the VLM rejects the
   top candidates** (e.g. decay to ≤30, or spend one more VLM check).
4. **Track-quality gate before deformation/friction/momentum verdicts**
   (mask-IoU continuity or DINOv2 self-similarity of the track; discard
   segments that jump). Kills the drift-as-physics class of FPs.
5. **VLM confirm prompts: ask "did this happen?" before "is it a
   violation?"** — show the crops and require the model to verify the claimed
   motion/overlap is visible, with an explicit "measurement may be a tracking
   artifact" escape hatch (gravity's prompt already has the pattern; port it).
6. **Momentum: exempt actuated agents** (arm/gripper/hand subjects named by
   the tracker) from transfer conservation, or require both bodies free.
7. **Collision: containment check** (mask-inside-mask ≠ interpenetration).
8. **Report: down-weight non-discriminative tests** — any test that fires
   ≥90 on >80% of a reference real set should be shrunk toward 0 weight in
   the aggregate score.

## Caveats

- AI-side numbers are from 3 clips — directional only. A matched re-run on
  the AI set after fixes 1–4 is the right next step.
- All VLM judgments here are Gemini 3.1 Pro via CreateAI; Flash or local Qwen
  may behave differently on the same prompts (s1_vlm's local token-prob path
  was not exercised in the main pass).
- Single run; VLM nondeterminism not averaged.

## Artifacts

- `eval_reports/2026-07-23/<video>/summary.json` — per-video roll-up
- `eval_reports/2026-07-23/<video>/<pipeline>.jsonl` — full event streams
  (the framework's own per-video diagnostic reports incl. `s4_report.jsonl`)
- `index.csv` — every (video, pipeline) row · `separation.csv` — per-pipeline
  real/AI means + AUC · `../2026-07-23.log` — run log

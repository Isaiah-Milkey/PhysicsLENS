# TODO — Rapidata human-agreement benchmark (week of 2026-07-30)

Owner: A4 · Requested by: A2 · Branch: `som-benchmark` (new, per A2's request)

## What was asked

From A2's messages, three things:

1. **Benchmark the current pipeline** using the batch runner A2 just shipped, and
   **find where it fails most** — that's this week's stated goal, not a leaderboard number.
2. **Check agreement with human sentiment** on
   [`Rapidata/sora-video-generation-physics-likert-scoring`](https://huggingface.co/datasets/Rapidata/sora-video-generation-physics-likert-scoring)
   (~198 Sora clips, crowd-rated 1–5 for physical plausibility).
3. **Write it as a re-runnable script**, because A2 and A1 are changing pipeline
   stages *this week* and want the same benchmark re-run next week to measure the delta.

Push to a new branch, like A1's `framework-eval`.

---

## Two pieces of this already exist — don't rebuild them

**(a) The Rapidata human-agreement eval, at the VLM level — mine, 2 weeks ago.**
`backend/scripts/vlm_rapidata_eval.py` (`0bebbcf`) already scores single VLMs against this
exact dataset and writes `vlm_rapidata_results.json`.

Current numbers, stratified n=60 of 198:

| model | Spearman | median-split AUC | logprob Spearman | logprob AUC |
|---|---|---|---|---|
| Qwen2.5-VL-7B | 0.297 | 0.644 | 0.343 | 0.700 |
| InternVL3-8B | 0.157 | 0.533 | **0.513** | **0.816** |

**So the question this week is not "does PhysicsLENS correlate with humans" — it's
"does the four-stage pipeline beat a single VLM prompt?"** A 7B VLM with one prompt
already reaches ρ≈0.51 / AUC≈0.82. If the full triage→localize→specialize→diagnose
pipeline lands below that, it is the single most important finding of the week and
A2 needs to hear it early, not in the writeup.

Reuse its scoring machinery (`spearman`, `median_split_auc`, `load_labels`,
`stratified_sample`) so old and new numbers stay directly comparable.

**(b) A full-pipeline eval harness — A1's, on `origin/framework-eval`.**
`backend/scripts/eval_framework.py` (233 lines) already runs every stage over a video set and
dumps `s1_*.jsonl`, `s2_*.jsonl`, `s3_*.jsonl`, `s4_report.jsonl` and `summary.json` per clip,
plus a `REPORT.md`. A1 ran it on the EWMBench **real**-video set as a 30-video false-positive
check.

That is exactly the "run the pipeline and persist everything" infrastructure this task needs.
**Point it at Rapidata and add human-agreement scoring on top — do not write a third harness.**
A2 explicitly pointed at this branch ("A1 published a branch... if that helps").

The two evals are complementary and worth reporting together:

| set | videos | what it measures |
|---|---|---|
| EWMBench (A1) | real | false-positive rate — does it flag physics that is genuinely fine? |
| Rapidata (this) | AI, continuous human label | rank agreement — does it order badness the way humans do? |

---

## Progress (2026-07-30)

**Done** — branch `som-benchmark` off `origin/framework-eval`, commit `1cef21b`:
- `rapidata_prepare.py` stages the Sora clips + a labels file, reusing
  `load_labels`/`stratified_sample` from `vlm_rapidata_eval.py` so the clip set is
  identical to the VLM benchmark.
- `eval_framework.py` gains `--videos-dir`, `--labels`, `--bootstrap` and a
  `human_agreement()` report. No second harness.
- Validated the analysis on synthetic summaries with a planted correlation:
  recovers rho=0.83 on the seeded pipeline, 0.13 with a CI spanning zero on a
  pure-noise one.
- Live-tested `s1_temporal`, `s1_optical_flow`, `s1_vlm` (API) on real clips.

**Fixed a blocker:** `.env` defined a differently-named key than `tools/createai.py`
(now `tools/llm_api.py`) read — the name it expected appeared nowhere in the codebase.
Every API call would have failed with a missing-credential error. Added the correct
name locally (`.env` is gitignored, so nothing to commit). **Worth telling the team**
— anyone else with the same `.env` hits this.

**Early smell, n=2 so do not over-read:** `s1_temporal` returned severity 100 on
both clips and `s1_vlm` returned 95 on both. A screen that saturates carries no
ranking signal regardless of how accurate it is. The per-pipeline table in
`human_agreement()` is built to show exactly this — check it on the real run.

---

## Tasks

### 1. Pin the baseline before anyone changes anything — do this first
- [ ] `git rev-parse HEAD` and record the SHA at the top of the results file.
      A2 and A1 are altering stages *while* this runs; without a pinned
      commit, next week's "improvement" is unattributable.
- [ ] Branch off it: `git checkout -b som-benchmark`
- [ ] Note GPU, `.env` model selection, and pipeline settings in the same header.

### 2. Decide the label→prediction mapping (blocking, needs a decision)
Human label is `LikertScoreNormalized` ∈ [0,1], **higher = more implausible**.
PhysicsLENS severity is also higher = worse, so the correlation should be **positive** —
worth asserting in code, a sign flip here silently inverts every conclusion.

The pipeline emits per-failure `severity` and `confidence` (`stage4/diagnostic_report.py`),
not a single scalar. Candidate reductions:
- [ ] `max_severity` across confirmed failures
- [ ] mean severity, weighted by confidence
- [ ] count of confirmed failures
- [ ] stage-1 raw suspicion score (comparable to the VLM baseline)
- [ ] stage-4 LLM-assigned overall score, if one exists

**Do not pick one up front.** Save the full diagnostic JSON per clip and compute all of
them offline — the expensive part is running the pipeline, and re-scoring is free.
This also means next week only the pipeline re-runs, not the analysis.

### 3. Scope the run against the runtime budget
A2 measured **~21 min for 4 clips ≈ 5.25 min/clip**.

| n clips | est. runtime |
|---|---|
| 20 | ~1.8 h |
| 60 (matches existing VLM baseline) | **~5.25 h** |
| 198 (full) | ~17.3 h |

- [ ] Use **n=60 with the same stratified sample** as `vlm_rapidata_eval.py` — that makes
      the pipeline directly comparable to the VLM baseline above, which is the whole point.
- [ ] Confirm whether the batch UI can run unattended overnight, or whether the headless
      `backend/scripts/run_pipeline.py` path is safer for a 5 h job.

### 4. Extend A1's `eval_framework.py` — don't write a new one
- [ ] Branch from `origin/framework-eval`, not `main`, so A1's harness comes along
- [ ] Add a Rapidata dataset source: `load_labels` + `stratified_sample` from my VLM script
- [ ] Add the human-agreement scoring pass (Spearman / median-split AUC) over A1's
      `summary.json` output — A1's harness already persists per-stage JSONL, runtime and
      GPU, so this is a scoring layer, not a rewrite
- [ ] Verify A1's per-clip dump already captures the diagnostic report text (A2's
      "Enable Diagnostic Report") — needed for §6
- [ ] Resumable: skip clips already in the output file. A 5 h job will get interrupted.
- [ ] `--n`, `--pipeline`, `--out` flags; results merge into a single JSON like the VLM script
- [ ] Report Spearman **and** median-split AUC for every candidate mapping from §2,
      side by side with the VLM baseline in the same table

### 5. Cross-check against A2's UI
- [ ] Run 3–4 of the same clips through the batch UI with Diagnostic Report on
- [ ] Export, and confirm the scores match the script's output for those clips
- [ ] If they diverge, the script is measuring something other than what the team sees —
      resolve before trusting any number

### 6. Failure analysis — this is the actual deliverable
A2: *"discover parts where the system fails most... let us know if there are certain
aspects we should add/change."* The correlation is one number; this is what drives next
week's changes.

- [ ] Rank clips by |human − predicted| and read the diagnostic reports for the worst ~10
- [ ] Split the error: **false alarms** (humans fine, tool flags) vs **misses**
      (humans object, tool clean). These imply opposite fixes.
- [ ] Attribute by stage: did stage 1 fail to flag it, or did stage 3 reject a real failure?
      Ranker picks the top-3 specialists — check whether the right specialist was even run.
- [ ] Break down by failure type (collision / gravity / momentum / friction / deformation /
      fluid / causality) — which specialists carry their weight and which never fire
- [ ] Record runtime and peak GPU per stage: if one stage costs 60 % of wall clock for no
      correlation gain, that is a finding

### 7. Report back
- [ ] Table: VLM baseline vs full pipeline, Spearman + AUC, same 60 clips
- [ ] Top failure modes with example clips and the diagnostic text
- [ ] Concrete add/change recommendations for next week
- [ ] PR from `som-benchmark`; keep the script committed so next week is one command

---

## Questions for A2

1. **Does the pipeline expose a single scalar plausibility score**, or should I derive one
   from severities? If there's an intended aggregate I'd rather use it than invent a mapping.
2. **Which pipeline id** is the current best/default for benchmarking — and does the
   Hypothesis Ranker top-3-specialist path count as "the system", or should I fix the
   specialist set so next week's comparison isn't confounded by a different ranker?
3. **n=60 or the full 198?** 60 matches the existing VLM baseline and fits in a night;
   198 is ~17 h.
4. **Are you and A1 changing stage 1?** If so, `vlm_rapidata_eval.py`'s numbers stop
   being a valid baseline and I should re-run it on the pinned commit too.
5. **Should A1's `framework-eval` script be the base** rather than a new one? Avoids
   two divergent benchmark harnesses.

---

## Risks

- **Moving target.** Stages change mid-week. Mitigation: pin the SHA (§1), and if a change
  lands mid-run, finish on the pinned commit rather than mixing versions.
- **~198 clips is a small, single-source sample.** All Sora, all one generation batch.
  Spearman on n=60 has a wide interval — report it, and don't over-read a 0.05 difference.
- **Human labels are crowd aggregates, not ground truth physics.** Disagreement can mean
  the tool is wrong *or* that raters weighted visual polish over physics. Worth reading a
  few diagnostic reports before concluding the tool is at fault.
- **Runtime.** 5.25 min/clip makes iteration slow. Resumability (§4) is not optional.

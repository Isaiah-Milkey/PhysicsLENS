# PhysicsLENS specialist evaluation — every test, every table, and why things fail

_Sept 5–11 2026. 68 scored conditions · 4 dataset variants · 8 judge models ·
~120,000 model calls. Every number below is reproducible from the named script._

**Reading the marks in the tables**

- `!` **dead** — ≥75 % of clips share one identical score, so the AUC is
  tie-breaking noise, not a measurement.
- `~` **flat** — values are distinct but the interdecile range is < 0.10, so the
  probe technically ranks but barely commits.

Cells carrying a mark should not be quoted on their own. Both diagnostics exist
because a probe can fail in two different ways and a tie-based check only catches
one of them.

---

## Contents

1. [Data](#1-data)
2. [The 15 experiments](#2-the-14-experiments) — table + conclusion each
3. [Why the weak specialists fail](#3-why-the-weak-specialists-fail)
4. [Is the evaluation itself wrong?](#4-is-the-evaluation-itself-wrong)
5. [Bugs found](#5-bugs-found-and-fixed)
6. [Final conclusion](#6-final-conclusion)

---

## 1. Data

| set | clips | annotated | role |
|---|---|---|---|
| cosmos3-nano | 80 | 80 | AI, generated from real start frames |
| Wan2.2-TI2V-5B | 64 | 64 | AI, same 64 source demos |
| real demonstrations | 82 | — | ground truth by construction |
| VideoPhy-2 | 1,200 | 704 | general physics, 7 generators |

64 source demos have a full real/cosmos/wan triplet. Every AI clip was
conditioned on a start frame from its real counterpart, so a real/AI pair shares
scene, objects, camera and task and differs only in whether the physics is
genuine — a control VideoPhy-2 cannot provide, because every clip there is AI.

Positives per specialist (RobotBench / VideoPhy-2): collision 76/167,
deformation 58/153, causality 53/—, momentum 23/100, gravity 18/93, fluid 12/60,
friction 7/35, permanence 0/21.

**Three different tests are run, and they answer different questions:**

| test | positives | negatives | note |
|---|---|---|---|
| attribution | this violation | a **different** violation | hardest; immune to "spot the AI" |
| detect_clean | this violation | AI clips annotated clean | honest detection, few negatives |
| detect_real | this violation | the real demonstrations | easiest, most confounded |

Unless stated, tables below are **attribution on RobotBench**.

---

## 2. The 15 experiments

### Test 1 — Probe wording

| condition | fluid | causality | gravity | momentum | deformation | collision | mean |
|---|---|---|---|---|---|---|---|
| `winners/f4` (VideoPhy-tuned) | 0.569! | 0.606! | 0.533~ | 0.500! | 0.605~ | 0.492 | 0.551 |
| **`robot/f4`** (annotator vocabulary) | 0.898! | 0.698 | 0.660~ | 0.557 | 0.532! | 0.452~ | **0.633** |
| `winners/qwen7b` | 0.862 | 0.607 | 0.369~ | 0.436~ | 0.498~ | 0.579 | 0.559 |
| `robot/qwen7b` | 0.904 | 0.446~ | 0.567 | 0.564~ | 0.475~ | 0.554~ | 0.585 |

**Conclusion.** The single biggest win of the week, **+0.082**. Rewriting the
probes from the annotators' own words — "moves on its own no contact" (×11),
"appears out of nowhere" (×9), "deforms/grew/morphs" (×20) — beat wording tuned
for ballistic physics. It did not merely improve fluid, it **revived** it: from
0.569 and dead to 0.898, the best cell in the study.

**Why the original wording failed.** VideoPhy-2 failures are loud (a ball through
a wall). Robot-manipulation failures are quiet (a gripper closing slightly wrong,
a cloth sliding untouched). Asked "does anything end up moving FASTER than before?"
about a robot wiping a table, the model correctly answers "no" on every clip — the
question is well-posed and simply never fires.

---

### Test 2 — Judge model (8 tried)

| condition | fluid | causality | gravity | momentum | deformation | collision | mean |
|---|---|---|---|---|---|---|---|
| `gemma4-31b` | 0.898! | **0.698** | 0.660~ | 0.557 | 0.532! | 0.452~ | 0.633 |
| `qwen2.5-vl-7b` | 0.904 | 0.446~ | 0.567 | 0.564~ | 0.475~ | 0.554~ | 0.585 |
| `qwen2.5-vl-32b` | 0.920 | 0.587 | 0.481 | 0.446 | **0.617** | 0.571~ | 0.604 |
| `internvl3-8b` | 0.911 | 0.424~ | **0.612** | 0.618~ | 0.534 | 0.475~ | 0.596 |
| `internvl3-14b` | 0.922 | 0.525~ | 0.590 | 0.537 | 0.519 | 0.528~ | 0.603 |
| `llama4-scout-17b` | **0.953** | 0.399~ | 0.487~ | **0.700**~ | 0.537~ | 0.458~ | 0.589 |
| `smolvlm2-2.2b` | 0.490~ | 0.568~ | 0.427~ | 0.505~ | 0.479~ | **0.625**~ | 0.516 |

Ruled out: **qwen3-vl-32b** (returns no logprobs), **gemma3-27b** and
**llama4-maverick** (reject images entirely).

**Conclusion.** Means cluster at 0.59–0.63 across a 2.2 B → 32 B range. Scale
buys almost nothing. The real differentiator is **coverage** — how many
specialists a judge keeps alive at all:

| judge | specialists alive |
|---|---|
| qwen2.5-vl-32b | 5/5 |
| internvl3-14b | 4/5 |
| internvl3-8b | 3/5 |
| qwen2.5-vl-7b | 2/5 |
| gemma4-31b | **1/5** |

**Why per-specialist routing does not work.** Three of four specialists pick the
**same** judge on both datasets (`internvl3-8b` every time). That is one model
being better overall, not per-specialist affinity. Stability confirms it: the
modal winner's win-rate is at coin-flip for fluid (38 %) and gravity (49 %).

---

### Test 3 — Frame count

| condition | fluid | causality | gravity | momentum | deformation | collision | mean |
|---|---|---|---|---|---|---|---|
| 2 frames | 0.811 | 0.501~ | 0.588 | 0.628~ | 0.559 | 0.526~ | 0.602 |
| 4 frames | 0.839 | 0.407~ | 0.519 | 0.657~ | 0.477~ | 0.505~ | 0.568 |
| 8 frames | 0.904 | 0.446~ | 0.567 | 0.564~ | 0.475~ | 0.554~ | 0.585 |
| **16 frames** | 0.913 | 0.525~ | 0.609 | 0.643~ | 0.553 | 0.582~ | **0.637** |

**Conclusion.** 8× more frames buys **+0.035**, and not monotonically — 2 frames
beats 4. If the model were integrating motion, temporal density would be the
dominant variable. It isn't.

---

### Test 4 — Caption content

| condition | fluid | causality | gravity | momentum | deformation | collision | mean |
|---|---|---|---|---|---|---|---|
| full (Scene + Action) | 0.898! | 0.698 | 0.660~ | 0.557 | 0.532! | 0.452~ | 0.633 |
| scene only | 0.894! | 0.665 | 0.648~ | 0.548 | 0.493! | 0.448~ | 0.616 |
| action only | 0.928~ | 0.721~ | 0.575~ | 0.534 | 0.503! | 0.481~ | 0.624 |
| **task only** | 0.931~ | **0.766**~ | 0.550 | 0.565 | 0.497! | 0.503~ | **0.635** |
| none | 0.932~ | 0.755 | 0.599 | 0.551 | 0.506! | 0.425~ | 0.628 |

**Conclusion.** Removing the caption entirely is as good as any version of it.
Causality *improves* without it (0.698 → 0.755): the long scene description is a
distractor, not context. A one-line task statement is the best of a flat field.

---

### Test 5 — Frame-order shuffle (the control)

| condition | fluid | causality | gravity | momentum | deformation | collision | mean |
|---|---|---|---|---|---|---|---|
| gemma4 ordered | 0.898! | 0.698 | 0.660~ | 0.557 | 0.532! | 0.452~ | 0.633 |
| gemma4 **shuffled** | 0.914! | 0.663 | 0.592 | 0.583 | 0.518! | 0.459~ | 0.622 |
| qwen7b ordered | 0.904 | 0.446~ | 0.567 | 0.564~ | 0.475~ | 0.554~ | 0.585 |
| qwen7b **shuffled** | 0.882 | 0.474~ | 0.534 | 0.516~ | 0.478~ | 0.507~ | 0.565 |

**Conclusion — the load-bearing negative result.** Randomising the frame order
costs ≈ **0.015**. Reproduced on three datasets and eight models. The specialists
are single-frame appearance detectors wearing physics vocabulary; they are not
reading motion. **This one fact explains most of the failures below.**

---

### Test 6 — Probe depth

| condition | fluid | causality | gravity | momentum | deformation | collision | mean |
|---|---|---|---|---|---|---|---|
| **1 question** (gemma4) | 0.898! | 0.698 | 0.660~ | 0.557 | 0.532! | 0.452~ | **0.633** |
| 32 sub-probes (gemma4) | 0.927 | 0.631 | 0.592 | 0.494 | 0.388 | 0.520 | 0.592 |
| 1 question (qwen7b) | 0.904 | 0.446~ | 0.567 | 0.564~ | 0.475~ | 0.554~ | 0.585 |
| 32 sub-probes (qwen7b) | 0.888 | 0.582 | 0.442 | 0.531 | 0.564 | 0.496 | 0.584 |

**Conclusion.** Five specific sub-questions per specialist, rank-fused, are no
better than one good question. Decomposition adds noise, not resolution — each
extra sub-probe is another chance to answer "nothing wrong".

---

### Test 7 — Prompt injection of Stage-1/2 evidence

| condition | fluid | causality | gravity | momentum | deformation | collision | mean Δ |
|---|---|---|---|---|---|---|---|
| gemma4/robot | +0.001 | −0.070 | −0.054 | +0.079 | −0.012 | +0.008 | −0.008 |
| qwen7b/robot | −0.030 | +0.067 | −0.028 | −0.000 | +0.007 | −0.002 | +0.002 |
| gemma4/winners | +0.100 | — | — | −0.019 | −0.052 | — | +0.010 |
| qwen7b/winners | −0.078 | +0.018 | +0.086 | +0.056 | +0.018 | −0.033 | +0.011 |
| **mean** | −0.002 | +0.005 | +0.001 | +0.029 | −0.010 | −0.009 | **+0.003** |

**95 % CI [−0.018, +0.024]. Not significant.**

**Manipulation check — the model definitely read it.** Only ~75 % of clips carry
a signal extreme enough to mention; the rest get a byte-identical prompt and act
as a built-in placebo group.

| | mean \|Δscore\| | % of scores that moved |
|---|---|---|
| clips **with** an evidence block | 0.053 | **100 %** |
| clips **without** one | **0.0000** | **0 %** |

**Conclusion.** The specialist reads the evidence, changes its answer
substantially, and the change is **uncorrelated with truth**. This is noise
injection dressed as information.

**Why it fails.** The evidence is numeric and relational ("many tracked points
disappear mid-frame"). Converting it to a sentence asks the model to re-ground a
measurement it cannot itself perform against pixels it cannot temporally
integrate. It has no way to check the claim, so it shifts its prior arbitrarily.

---

### Test 8 — Motion rendered into the frames

| condition | fluid | causality | gravity | momentum | deformation | collision | mean |
|---|---|---|---|---|---|---|---|
| plain / gemma4 | 0.898! | 0.698 | 0.660~ | 0.557 | 0.532! | 0.452~ | 0.633 |
| plain / qwen7b | 0.904 | 0.446~ | 0.567 | 0.564~ | 0.475~ | 0.554~ | 0.585 |
| **trails / qwen7b** | 0.864 | 0.577~ | 0.575 | 0.623~ | **0.624**~ | **0.583**~ | **0.641** |
| diff / qwen7b | 0.858 | 0.544~ | 0.510 | **0.664**~ | 0.478~ | 0.540~ | 0.599 |
| trails / gemma4 | 0.912! | 0.686~ | 0.604~ | 0.545 | 0.513! | 0.468~ | 0.621 |
| diff / internvl8b | 0.835 | 0.431~ | 0.527 | 0.579~ | 0.477 | 0.461~ | 0.552 |

`trails` = camera-compensated Lucas-Kanade tracks drawn as fading tails.
`diff` = red overlay where the frame differs from ~0.4 s earlier.

**Conclusion.** The only intervention that moved the stuck specialists
(deformation 0.475 → 0.624, causality 0.446 → 0.577). It is also the only
intervention that changed **what the model sees** rather than rearranging what it
already said. **But it did not replicate on VideoPhy-2** — plain frames win every
specialist there — so treat it as robot-specific or noise until reproduced.

---

### Test 9 — Judge ensembling

| specialist | gemma4 | qwen7b | qwen32b | internvl8b | internvl14b | **ENSEMBLE** | best single |
|---|---|---|---|---|---|---|---|
| fluid | 0.898 | 0.904 | 0.920 | 0.911 | 0.922 | **0.947** | 0.922 |
| causality | 0.698 | 0.446 | 0.587 | 0.424 | 0.525 | 0.550 | 0.698 |
| gravity | 0.660 | 0.567 | 0.481 | 0.612 | 0.590 | 0.613 | 0.660 |
| momentum | 0.557 | 0.564 | 0.446 | 0.618 | 0.537 | 0.572 | 0.618 |
| deformation | 0.532 | 0.475 | 0.617 | 0.534 | 0.519 | 0.443 | 0.617 |
| collision | 0.452 | 0.554 | 0.571 | 0.475 | 0.528 | 0.516 | 0.571 |
| **mean** | | | | | | **0.607** | 0.681 |

**Conclusion.** Ensembling gains only on fluid. Mean 0.607 vs 0.604 for an
average single judge — nothing.

**Why it fails.** Judges agree with **each other** at 0.43 and with **humans** at
0.18. Averaging correlated errors cancels nothing; it just averages the same
mistake.

---

### Test 10 — Learned fusion (ridge, 33 features, nested CV)

| specialist | pos | best probe | signals only | probes only | ALL | gain |
|---|---|---|---|---|---|---|
| fluid | 12 | 0.922 | 0.732 | 0.913 | 0.893 | −0.029 |
| causality | 53 | 0.698 | 0.481 | 0.627 | 0.529 | −0.169 |
| gravity | 18 | 0.660 | 0.541 | 0.484 | 0.510 | −0.150 |
| momentum | 23 | 0.618 | 0.535 | 0.558 | 0.580 | −0.038 |
| deformation | 58 | 0.617 | 0.592 | 0.507 | 0.574 | −0.043 |
| collision | 76 | 0.571 | 0.588 | 0.540 | 0.612 | **+0.042** |
| **mean gain** | | | | | | **−0.065** |

**Conclusion.** Learning the combination is *worse* than not learning it. Only
collision — the largest category at 76 positives — gains.

**Why it fails.** 33 features against 12–77 positives overfits even with nested
cross-validation and alpha chosen on inner folds only. There is not enough
labelled data to fit anything.

---

### Test 11 — Numeric fusion of Stage-1/2 signals

Reported as a **pass rate across 10 probe configurations**, not a best case:
taking the max makes all six look significant, the median makes two.

| specialist | best signal | median fused | null p95 | configs passing | verdict |
|---|---|---|---|---|---|
| fluid | `s2_n_tracks` | 0.928 | 0.723 | **10/10** | **REAL** |
| causality | `s2_inner_death_frac` | 0.663 | 0.640 | **6/10** | **REAL** |
| momentum | `sp_gravity_flat` | 0.658 | 0.676 | 3/11 | config-dependent |
| collision | `s1_flow_entropy` | 0.602 | 0.638 | 3/10 | config-dependent |
| gravity | `s1_flow_entropy` | 0.671 | 0.701 | 2/10 | no |
| deformation | `s1_cam_motion` | 0.614 | 0.645 | 1/11 | no |

`null p95` repeats the whole best-of-28-signals search on shuffled labels, so it
is the bar a *selected* signal must clear. It sits at 0.63–0.73 — selection alone
reaches that.

**Conclusion.** Upstream evidence helps only when fused as **numbers**, never as
words (contrast Test 7), and only for the two specialists whose failures are
**geometric** rather than temporal: fluid wants track count and jerk (liquid
snapping between shapes); causality wants tracked points dying mid-frame away
from any border — objects appearing or vanishing untouched.

**Architectural consequence.** Do not build a prompt-passing bus from Stage 2 to
Stage 3. Have Stage 2 publish numeric signals and rank-fuse them after the
specialist runs. Two wires are worth adding; four are not.

---

### Test 12 — Critic pass (second verification stage)

| specialist | verify | evidence | refute | alternative | probe baseline |
|---|---|---|---|---|---|
| fluid | **0.945** | 0.936 | 0.906 | 0.872 | 0.898 |
| causality | 0.715 | 0.683 | 0.652 | 0.689 | 0.698 |
| gravity | **0.711** | 0.633 | 0.652 | 0.615 | 0.660 |
| momentum | 0.531 | 0.540 | 0.561 | 0.589 | 0.557 |
| deformation | **0.441 (sig −)** | 0.561 | 0.512 | 0.500 | 0.532 |
| collision | 0.462 | 0.487 | 0.416 | 0.394 | 0.452 |

**Mean over 24 specialist × style cells: −0.008. Zero significantly positive,
one significantly negative.**

**Conclusion.** A second verification pass does not rescue the weak specialists.

**Why it fails — and the easy explanation is ruled out.** Probe↔critic rank
correlation is only **0.22**, so the critic is *not* merely restating the probe:
it reasons independently. It is independently **wrong**. Asking a model that
cannot see motion to double-check a motion judgement produces a second blind
opinion, not a correction.

---

### Test 13 — Config strategy (split-half held-out)

| strategy | attribution | detect_clean |
|---|---|---|
| per-specialist tuning (best of ~15–29 each) | 0.611 | **0.650** |
| global (one config chosen per split) | 0.604 | 0.586 |
| **fixed config, chosen by nobody** | **0.620** | 0.605 |

**Conclusion.** A configuration nobody tuned beats per-specialist tuning out of
sample on the hard test. Selection bias grew **+0.066 → +0.100** as the search
went from 20 to 40 conditions while held-out mean did **not** improve — the
search saturated and further conditions bought only self-flattery.

---

### Test 14 — Cross-dataset transfer (the only leak-free test)

Configuration chosen on one dataset, scored on the other. The evaluation clips
played no part in the choice, so **no selection bias is possible**.

| specialist | VideoPhy→Robot | Robot→VideoPhy | in-sample best | worst drop |
|---|---|---|---|---|
| **fluid** | **0.917** | **0.910** | 0.953 / 0.910 | −0.036 |
| collision | 0.579 | 0.501 | 0.579 / 0.603 | −0.102 |
| gravity | 0.503 | 0.592 | 0.612 / 0.610 | −0.109 |
| momentum | 0.561 | 0.551 | 0.636 / **0.727** | **−0.176** |
| deformation | 0.534 | 0.544 | 0.629 / 0.589 | −0.095 |
| **mean transferred** | | | | **0.619** |

**Conclusion — the number to quote.** One specialist generalises: **fluid, 0.91
in both directions**, across two datasets with different physics, generators and
failure styles. Momentum is the cautionary case: its 0.727 in-sample on
VideoPhy-2 is the second-best figure in the project, and a configuration chosen
on robot data scores **0.551** on those same clips. That was a property of the
configuration, not of the specialist.

Causality could not be transfer-tested — VideoPhy-2 carries no causality
annotations. Within robot data it reaches 0.743 held-out.


### Test 15 — Forced-choice (one MCQ instead of eight probes)

Instead of eight independent probes, one question: *"Which ONE of these best
describes the main physics problem?"* with all eight specialists plus a **none**
option. Option order is permuted per clip. Judge: `internvl3-8b`, 8 frames.

**15a. Attribution — P(option) vs the same labels**

| specialist | pos | neg | MCQ AUC [95% CI] | 8 separate probes | delta |
|---|---|---|---|---|---|
| fluid | 12 | 112 | 0.888 [0.79, 0.96] | 0.911 | −0.024 |
| momentum | 23 | 101 | 0.653~ [0.54, 0.75] | 0.618 | +0.035 |
| gravity | 18 | 106 | 0.636 [0.47, 0.79] | 0.612 | +0.024 |
| deformation | 58 | 66 | 0.613 [0.51, 0.71] | 0.534 | **+0.079** |
| causality | 53 | 71 | 0.577~ [0.48, 0.68] | 0.424 | **+0.153** |
| collision | 76 | 48 | 0.555 [0.45, 0.66] | 0.475 | **+0.080** |
| **mean** | | | **0.654** | 0.596 | **+0.058** |

**Conclusion.** The second-largest gain of the week, behind only probe rewording
(+0.082) — and it lands precisely where predicted. The two probes diagnosed as
broken in §3.3 are the two that improve most: **collision** (saturated at 0.97 on
everything) +0.080, **deformation** (dead at 0.000) +0.079, plus causality +0.153.
Forcing the options to compete makes "everything is a collision failure"
inexpressible, because claiming one option costs mass that must come from
another. It is also ~8x cheaper: one call per clip instead of eight.

**15b. Argmax — can it actually pick the right answer?**

| measure | value |
|---|---|
| argmax matches a true label | **29.8 %** (37/124) |
| chance — always answer "collision", the commonest label | **61.3 %** |
| uniform random over 9 options | 11.1 % |

| | friction | collision | none | deformation | fluid | momentum | gravity |
|---|---|---|---|---|---|---|---|
| **it picks** | **56** | 36 | 15 | 8 | 6 | 3 | 0 |
| **truth** | **7** | 76 | 20 | 58 | 12 | 23 | 18 |

**Conclusion — the decision is worse than chance.** It answers *friction* 56
times when friction is the true label 7 times, and *never* answers gravity
though gravity is true 18 times. A constant "collision" would score 61 %; this
scores 30 %.

**Why.** The friction option reads *"something slides when it should grip, or
keeps sliding with nothing pushing it"*. In robot manipulation almost every clip
contains sliding — wiping, pushing, dragging — so the model selects the option
that is **descriptively apt**, not the one that is a **violation**. Forced choice
removes the saturation pathology but replaces it with a surface-plausibility
pathology.

**15c. The `none` option — does not work**

| test | AUC |
|---|---|
| 1 − P(none) vs clean AI clips | **0.436** (below chance) |
| 1 − P(none) vs real demonstrations | 0.580 |

| | mean P(none) |
|---|---|
| broken clips | 0.160 |
| clean AI clips | 0.136 |
| real demonstrations | 0.195 |

**Conclusion.** P(none) is *higher* on broken clips than on clean ones — the
model is slightly worse than random at recognising that nothing is wrong. Only
against real footage does it weakly separate, and that is the confounded test.

**15d. Position-bias control — passed**

| | A | B | C | D | E | F | G | H | I |
|---|---|---|---|---|---|---|---|---|---|
| mean P by printed position | 0.107 | 0.135 | 0.114 | 0.126 | 0.112 | 0.115 | 0.083 | 0.092 | 0.116 |

correlation(printed position, P(option)) = **−0.044**

**Conclusion.** No meaningful list-order effect; the per-clip shuffling worked
and the gain in 15a is not an ordering artefact.

**Overall.** Forced choice is the right framing for **ranking** — use P(option)
as a continuous score and it beats eight independent probes by +0.058, fixing
exactly the two saturated probes. It is the wrong framing for **deciding** — its
single answer is worse than a constant guess, and its abstain option is
anti-correlated with truth. Adopt the probabilities; discard the argmax.

---


**15e. All models, both datasets — the final MCQ grid**

_RobotBench (124 annotated clips), attribution AUC from P(option):_

| model | f | fluid | causality | gravity | momentum | deformation | collision | **mean** |
|---|---|---|---|---|---|---|---|---|
| **internvl3-8b** | 8 | 0.888 | 0.577~ | **0.636** | **0.653**~ | 0.613 | **0.555** | **0.654** |
| qwen2.5-vl-7b | 8 | 0.907 | **0.632** | 0.579 | 0.548~ | 0.552 | 0.471 | 0.615 |
| internvl3-14b | 8 | 0.877~ | 0.594~ | 0.568~ | 0.476~ | 0.598~ | 0.504 | 0.603 |
| qwen2.5-vl-32b | 8 | 0.903 | 0.512~ | 0.575~ | 0.498~ | 0.586 | 0.499 | 0.596 |
| gemma4-31b-it | 4 | 0.714~ | **0.710** | 0.554~ | 0.520~ | 0.477~ | 0.555 | 0.588 |

_VideoPhy-2 (704 annotated clips):_

| model | f | fluid | gravity | momentum | deformation | collision | permanence | friction | **mean** |
|---|---|---|---|---|---|---|---|---|---|
| **internvl3-8b** | 8 | 0.912 | 0.605 | **0.671** | 0.548 | 0.484~ | 0.597~ | **0.717** | **0.648** |
| qwen2.5-vl-7b | 8 | **0.936** | 0.593 | 0.668 | 0.544 | 0.530 | 0.621 | 0.638 | 0.647 |
| internvl3-14b | 8 | 0.925 | 0.549 | 0.613 | 0.568 | 0.529 | 0.609 | 0.606 | 0.628 |
| gemma4-31b-it | 4 | 0.813~ | **0.627** | 0.593~ | 0.571 | 0.551 | 0.621 | 0.588~ | 0.623 |
| qwen2.5-vl-32b | 8 | 0.770~ | 0.584 | 0.587~ | 0.576 | 0.551 | **0.705**~ | 0.577~ | 0.621 |

_Argmax accuracy, both datasets, against the majority-class baseline:_

| model | RobotBench | VideoPhy-2 | over-picks |
|---|---|---|---|
| gemma4-31b-it | 59.7 % (−1.6) | 16.8 % (−7.0) | collision (106 / 259) |
| internvl3-14b | 46.0 % (−15.3) | — | collision (84), friction (27) |
| internvl3-8b | 29.8 % (−31.5) | 13.8 % (−9.9) | friction (56), none (295) |
| qwen2.5-vl-7b | 11.3 % (−50.0) | 8.8 % (−14.9) | **none (93 / 480)** |
| qwen2.5-vl-32b | 7.3 % (−54.0) | — | **none (106)** |

baseline = always answer the commonest label: 61.3 % on RobotBench, 23.7 % on VideoPhy-2.

**Conclusion.**

1. **`internvl3-8b` wins both datasets** (0.654 / 0.648). This is the third
   independent line of evidence for the same judge — it also won the
   cross-dataset agreement test (Test 2) and the fixed-config comparison
   (Test 13). It is now a settled recommendation, not a per-dataset fluke.
2. **Scale hurts, in both families.** internvl3-8b 0.654 > internvl3-14b 0.603;
   qwen-7b 0.615 > qwen-32b 0.596. Same clips, same prompt, more parameters,
   worse result.
3. **The MCQ gain replicates across datasets** (0.654 robot / 0.648 VideoPhy-2
   vs an 8-probe baseline of 0.596). Unlike the motion-frame result in Test 8,
   this one transfers.
4. **Two opposite miscalibrations hide behind similar AUCs.** gemma4 and
   internvl3-14b *over-allege* (gemma4 answers collision on 106/124 clips);
   both qwen models *over-abstain* (qwen-32b answers "none" on 106/124,
   qwen-7b on 480/704). Neither is visible in the AUC ranking, and each would
   break a deployed system in the opposite direction.
5. **No model beats the majority baseline on argmax, on either dataset.** The
   closest (gemma4, −1.6 %) gets there only by effectively *being* the majority
   guess. Use `P(option)` as a score; discard `argmax`; the abstain option is
   unusable in both directions.

**Caveat on the VideoPhy-2 half.** Those labels come from the keyword taxonomy,
not human annotation, and 32 % of its rule text goes unmatched — so that side is
noisier than RobotBench's human categories.

---

## 3. Why the weak specialists fail

Four distinct causes. They are not the same for every specialist.

### 3.1 The failing categories almost never occur alone

| specialist | positives | appears **alone** | mean co-occurrence | AUC |
|---|---|---|---|---|
| fluid | 12 | **33 %** | 20 % | **0.914** |
| deformation | 58 | 24 % | 24 % | 0.539 |
| causality | 53 | 13 % | 27 % | **0.743** |
| collision | 76 | 13 % | 26 % | 0.540 |
| gravity | 18 | **6 %** | 33 % | 0.548 |
| momentum | 23 | **4 %** | 33 % | 0.556 |

correlation(solo-rate, AUC) = **+0.67** · correlation(co-occurrence, AUC) = **−0.62**

**69 % of clips carry two or more labels.** Pairwise, collision co-occurs with
deformation **64 %** of the time, gravity with momentum 50 %, momentum with
collision 52 %.

When two failure types nearly always appear together, **no detector can separate
them even in principle** — the distinction is not visually realised in the clips.
The attribution test asks "is this a collision failure or a *different* failure",
and for these categories that question has no stable answer. Gravity and momentum
appear without a co-label in 6 % and 4 % of their clips respectively.

### 3.2 The labels are annotator-dependent

| specialist | min usage | max usage | spread |
|---|---|---|---|
| fluid | 0.0 % | 13.3 % | **13.3 %** |
| gravity | 0.0 % | 13.1 % | 13.1 % |
| momentum | 0.0 % | 20.8 % | 20.8 % |
| collision | 20.0 % | 48.3 % | 28.3 % |
| causality | 3.3 % | 34.4 % | 31.1 % |
| deformation | 4.2 % | 37.7 % | **33.5 %** |

One annotator labels "Contact" 48 % of the time and "Collision" *never*; another
uses both. Causality usage runs 3 % to 34 % — a tenfold spread. Labels-per-clip
runs 1.77 to 2.60.

Because each annotator saw a **disjoint** set of clips, personal style and real
content differences are confounded and cannot be separated. A tenfold spread on
one category is nonetheless unlikely to be all content. **This is why ~150
double-annotated clips matter**: without them there is no ceiling, and model
scores are being compared to a target of unmeasured reliability.

### 3.3 Two specialists are broken at the probe level, not the concept level

Mean probe score on positives vs negatives (`robot/f4`):

| specialist | mean(pos) | mean(neg) | separation |
|---|---|---|---|
| causality | 0.231 | 0.098 | **+0.133** |
| momentum | 0.532 | 0.417 | +0.116 |
| fluid | 0.128 | 0.035 | +0.093 |
| gravity | 0.122 | 0.086 | +0.036 |
| collision | **0.970** | **0.938** | +0.032 |
| deformation | **0.000** | **0.010** | **−0.010** |

Two specific, *fixable* pathologies:

- **Collision saturates at the ceiling** — it answers ≈0.97 on everything. "Do
  the gripper and the object overlap?" is true of nearly every manipulation
  frame, because grippers touch things. The question is not wrong, it is
  non-discriminative.
- **Deformation saturates at the floor** — 0.000 on positives, and it scores
  negatives *higher*. The probe is dead and slightly inverted.

Both are wording problems. The same class of fix took fluid from 0.569 (dead) to
0.898.

### 3.4 The unfixable one: gravity and momentum are defined by rates

Gravity's positive/negative separation is **+0.036** — and this is *not* a
saturation artefact: the probe responds, it simply responds almost identically to
both classes.

That is Test 5 cashing out. "Does it fall at constant speed instead of
accelerating?" requires comparing the gap an object moves between *early* frames
with the gap between *late* frames. Shuffling the frames costs 0.015, so the
model is not performing that comparison. **Fluid works because a frozen splash or
a rigid blob is wrong in a single frame. Gravity, momentum and friction are
defined as rates across time — a quantity the model never computes.** No prompt
reaches a quantity that is never computed.

### 3.5 Summary of causes

| specialist | dominant cause | fixable? |
|---|---|---|
| collision | probe saturates at ceiling (0.97 everywhere) + 64 % co-occurrence with deformation | wording: **yes**; label overlap: no |
| deformation | probe dead and inverted (0.000) + 64 % co-occurrence with collision | wording: **yes**; label overlap: no |
| gravity | requires cross-frame rates; 6 % solo; 18 positives | **no** — architectural |
| momentum | requires cross-frame rates; 4 % solo; 23 positives | **no** — architectural |

Two different failures wearing the same mask.

---

## 4. Is the evaluation itself wrong?

Audited for the three ways it could be.

### 4.1 Coverage — complete

| | count |
|---|---|
| staged total | 226 |
| real (excluded from attribution) | 82 |
| AI clips | **144** (80 cosmos + 64 wan) |
| AI with a violation → used by attribution | **124** |
| AI annotated clean | 20 |
| **scored by the probe file** | **124 / 124, zero missing** |

Every specialist is evaluated on **all 124** clips; only the positive/negative
split changes (fluid 12+112 … collision 76+48). One clip is genuinely absent: of
145 annotated rows, one cosmos row had a blank rating and was dropped at staging.

### 4.2 Generator confounding — checked, and clean

Wan is rated worse than cosmos (1.84 vs 2.33) and category rates do differ by
generator (causality +19 % in wan, deformation −21 %), so the concern was
legitimate. But the probe scores **cannot tell the generators apart**:

| specialist | AUC predicting wan-vs-cosmos |
|---|---|
| fluid | 0.520 |
| causality | 0.511 |
| gravity | 0.485 |
| deformation | 0.467 |
| collision | 0.463 |
| momentum | 0.369 |

Decisive test — AUC computed **within** each generator separately:

| specialist | pooled | cosmos only | wan only | within-avg | inflation |
|---|---|---|---|---|---|
| fluid | 0.898 | 0.967 | 0.849 | 0.908 | −0.010 |
| causality | 0.698 | 0.697 | 0.726 | 0.711 | −0.013 |
| gravity | 0.660 | 0.709 | 0.602 | 0.655 | +0.005 |
| momentum | 0.557 | 0.501 | 0.608 | 0.555 | +0.002 |
| deformation | 0.532 | 0.469 | 0.593 | 0.531 | +0.001 |
| collision | 0.452 | 0.391 | 0.527 | 0.459 | −0.007 |
| **mean** | **0.633** | | | **0.637** | **−0.004** |

Pooling contributes **−0.004**. If the specialists were exploiting the generator
difference, pooled would sit well above within-generator. It sits marginally
below.

### 4.3 What *is* weak about the evaluation

The weaknesses are in the **labels**, not the arithmetic:

1. **Negatives are contaminated by construction** (§3.1) — collision's negatives
   are clips with a different violation, but collision and deformation co-occur
   64 % of the time.
2. **69 % of clips are multi-label**, and gravity/momentum are almost never solo.
3. **No second annotator**, so there is no ceiling, and per-category usage varies
   up to tenfold between annotators (§3.2).
4. **Sample sizes**: friction 7, fluid 12, gravity 18, momentum 23 positives are
   all below the ~40 where a 95 % CI stops spanning chance.

**Verdict: the pipeline measures what it claims to, on all the data. The ceiling
it measures against is the soft part.**

---

## 5. Bugs found and fixed

Each would have produced confident, wrong numbers.

1. **Sampling artefact inverted a result.** Stage-1/2 signals sampled a fixed 48
   frames per clip, but real demos run ~12 s @ 30 fps and AI clips 5.04 s @ 24 fps
   — AI samples sat 0.10 s apart, real ones 0.26 s. Every velocity and jerk
   statistic came out ~2.5× smaller for AI *purely from sampling*; it "found"
   real video was jerkier at AUC 0.83. On a fixed wall-clock grid the result
   **inverts** to AI jerkier at 0.819. Real fps in this corpus spans 5–50.
2. **Gateway image cap, silent.** `gemma4-31b-it` now rejects >4 images
   ("At most 4 image(s) may be provided in one prompt"), *intermittently*,
   because the model group load-balances and only some backends enforce it. Six
   conditions wrote 0-cell files and exited 0. `domain_probes.py` now exits
   non-zero below 50 % cell coverage. **Affects any pipeline sending >4 frames to
   gemma4, including Stage-1 VLM suspicion.**
3. **Tied-rank AUC.** A constant probe scored 0.000 instead of 0.500 because ties
   were not averaged. Affected the new scorer only.
4. **Caption truncation deleted the useful half.** Staging capped captions at 400
   chars and the prompt again at 200, so the `Action:` line reached the model on
   54 of 226 clips.
5. **Provenance mislabelled.** `domain_probes.py` wrote `"model": MODEL` (module
   default), so non-default runs recorded the wrong judge in their JSON.
6. **Second selection layer.** The Test-11 table originally took the *max* across
   10 fusion runs, making all six specialists look significant; pass-rate
   reporting drops it to two.
7. **Apples-to-oranges judge means.** Judges cover different specialists, so a
   mean over each judge's own subset compared different tasks — a judge surviving
   only on the two easy specialists "won" by never being tested on the hard ones.

---

## 6. Final conclusion

**Fourteen experiments, 68 conditions, ~120,000 model calls. One specialist
works.**

**Fluid is real and ready** — 0.91 transferred in both directions across two
datasets with different physics, generators and failure styles, and it is the one
specialist that also gains from Stage-2 signals (10/10 configurations).
**Causality is promising** at 0.743 held-out, but only robot data supports it.
**The other four sit at chance once they have to generalise.**

**Only two interventions produced replicated gains**: probe rewording (+0.082)
and numeric signal fusion for fluid and causality. Judges, frames, captions,
batteries, prompt injection, ensembling, learned fusion, per-specialist tuning,
critics and motion rendering were flat, negative, or failed to survive a change
of dataset.

**The reason is Test 5.** Shuffling frames costs ~0.015, on three datasets and
eight models. These specialists are single-frame appearance detectors wearing
physics vocabulary. Once that is accepted, the twelve failures stop looking like
twelve separate disappointments and start looking like **one fact confirmed
twelve ways: no amount of rearranging a blind judge's answers restores sight.**

The label problems (§3.1, §3.2) sit *on top* of that: even a model that could see
motion would be capped by categories that co-occur 64 % of the time and are
applied tenfold differently by different annotators.

### Recommended next steps, in order

1. **Ship fluid and causality.** One judge (`internvl3-8b`), robot-tuned wording,
   4 frames, caption optional. Wire `s2_n_tracks` → fluid and
   `s2_inner_death_frac` → causality as numeric rank-fusion, not prompt text.
2. **Stop tuning per specialist.** A single fixed config beat per-specialist
   selection out of sample (0.620 vs 0.611), and the gap widens as the search
   grows.
3. **One targeted rewording pass on collision and deformation.** Their probes are
   saturated at ceiling and floor respectively (§3.3) — the same class of fix
   took fluid from dead to best-in-study. ~20 minutes of compute.
4. **Fix the annotation before re-testing anything.** Single-label-primary (or a
   ranked primary category), ~150 double-annotated clips for a ceiling, and
   `t_start`/`t_end` timestamps — the last of which would allow within-clip
   negatives, the only design holding scene, objects and generator constant.
5. **For gravity, momentum and friction, a frame-sampling VLM is the wrong
   instrument.** Either a video encoder with real temporal modelling for Stage 3,
   or score those specialists from the numeric signals directly and reserve the
   VLM for where it demonstrably works.

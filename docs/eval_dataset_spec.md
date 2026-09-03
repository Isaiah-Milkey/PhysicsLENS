# Annotation spec for the PhysicsLENS evaluation dataset

What we need your annotators to record, why each field exists, and what breaks if it
is missing. Every claim below is backed by a measurement we already ran on VideoPhy-2
(1,200 clips) — the numbers are quoted inline so you can see what each field buys.

Hand the harness a folder of videos plus one JSON/CSV of labels:

```bash
python backend/scripts/eval_prepare.py --videos vids/ --labels labels.csv \
    --out data/ourbench --inspect        # prints what it detected, extracts nothing
python backend/scripts/eval_prepare.py --videos vids/ --labels labels.csv \
    --out data/ourbench                  # stage it
python backend/scripts/eval_run.py --data data/ourbench
```

---

## 1. The annotation form, in priority order

| # | Field | Required? | What it unlocks | What it costs the annotator |
|---|-------|-----------|-----------------|------------------------------|
| 1 | `id` | **hard requirement** | maps a label row to a video file | free |
| 2 | `rating` | **hard requirement** | whole-clip physics quality; every correlation/AUC we report | ~5 s, one slider |
| 3 | `has_violation` | **the biggest gap in VideoPhy-2** | separates "detect any failure" from "attribute it to the right specialist" | 1 checkbox |
| 4 | `category` | needed for per-specialist numbers | routes a clip to the gravity/collision/… specialist without our keyword guesser | 1 dropdown, multi-select |
| 5 | `t_start`, `t_end` | **highest-value new field** | the only way to test whether a model reads motion at all | ~15 s, scrub to the moment |
| 6 | `rules` | strongly recommended | free-text failure description; feeds retrieval + audits `category` | ~20 s typing |
| 7 | `object` | recommended | object-conditioned questions; joins to our SAM3 tracker output | 1 short string |
| 8 | `severity` | recommended | lets us check we score *magnitude*, not just presence | 1 dropdown |
| 9 | `caption` | recommended | intended content; also a shortcut we must be able to ablate | free (it's the generation prompt) |
| 10 | `generator` | recommended | per-generator breakdown, distribution-shift check | free (metadata) |
| 11 | `rating2` | on a ~150-clip subset | inter-annotator agreement = the ceiling we are allowed to claim | doubles cost on that subset only |

Fields 1–5 are the ones that change what we can *measure*. Fields 6–11 change how
well we can *explain* the result.

---

## 2. Why each field, with the measurement behind it

### 2.1 `has_violation` — the flag VideoPhy-2 never gave us

VideoPhy-2 only wrote rule text for clips that **failed**. So when we evaluate the
gravity specialist, its negatives are clips with a *different* violation, not clean
clips. That is a defensible test — it is what forces the 0.5 floor to mean something —
but it means **we have never measured the easy case**: clean video vs broken video.

With a clean/broken flag we can report two separate, honestly-labelled numbers:

| Test | Positives | Negatives | What a good score proves |
|------|-----------|-----------|--------------------------|
| **Detection** | clips with a gravity violation | clips with *no* violation | the specialist notices something is wrong |
| **Attribution** | clips with a gravity violation | clips with a *different* violation | the specialist knows *which* thing is wrong |

Right now we can only run the second, and we quote 0.586–0.887 for it. A reviewer's
first question will be "what about clean clips?" and today the answer is "we don't
have any." Please label clean clips explicitly — they need to be a real fraction of
the set, not an afterthought. Target **25–35 % clean**.

### 2.2 `t_start` / `t_end` — the field that fixes our worst finding

Our strongest single result is also our most uncomfortable one. We shuffled the 8
frames of every clip into a random order and re-ran the specialists:

| Specialist | frames in order | frames shuffled | Δ |
|------------|-----------------|-----------------|---|
| gravity | 0.664 | 0.665 | +0.001 |
| permanence | 0.649 | 0.648 | −0.001 |
| friction | 0.696 | 0.706 | +0.010 |
| deformation | 0.606 | 0.609 | +0.003 |
| collision | 0.618 | 0.598 | −0.020 |
| momentum | 0.586 | 0.550 | −0.036 |

Five of six specialists score the same on scrambled frames. Whatever they are
detecting, it is not motion — it is "does any single frame look wrong". We cannot fix
this by prompting, and we cannot even *diagnose* it further with clip-level labels,
because a clip-level label gives the model no reason to look at a specific moment.

Timestamps fix this in two ways:

1. **Localization accuracy becomes measurable.** Ask the model *when* it went wrong and
   score against `[t_start, t_end]`. A model that gets the clip right and the moment
   wrong is doing appearance matching, and we can finally prove it.
2. **Within-clip negatives become possible.** Take the window `[t_start, t_end]` as a
   positive and a window from the *same clip* far from it as a negative. Same scene,
   same objects, same caption, same generator, same lighting — every appearance
   shortcut is held constant, and the only remaining difference is what happens over
   time. No other design removes those confounds.

Precision needed: **±0.25 s is plenty.** If a failure is diffuse ("the whole clip
drifts"), let annotators tick a `diffuse` box rather than inventing a boundary.

### 2.3 `category` — stop making us guess from free text

Today `rule_taxonomy.py` keyword-matches free-text rules onto our 7 categories. It is
deterministic and auditable, but it is a guess, and it silently drops rules whose
wording doesn't match. A dropdown removes that whole error source.

Use exactly these seven, and allow **multi-select** — real failures are often two
things at once ("the ball passes through the wall and then floats"):

`collision` · `gravity` · `momentum` · `friction` · `deformation` · `fluid` · `permanence`

Add an **`other`** option with a mandatory free-text box. If `other` exceeds ~10 % of
failures, our taxonomy is wrong and we should know that before we build around it.

### 2.4 `rules` — keep it even though `category` exists

The free text is what the retrieval channel embeds, it is how we audit whether the
dropdown was used consistently, and it is the only record of *what specifically* broke.
Ask for one concrete sentence naming the object and the physical expectation
("the mug passes through the table instead of resting on it"), not a category name
restated.

### 2.5 `object` — joins annotation to our tracker

We already run SAM3 / LK tracking and can extract per-object trajectories. Knowing
that the failure belongs to "the red ball" and not "the person" lets a specialist ask
about the right object instead of the whole frame, and lets us check whether our
object-level physics fits (`physics_rules.py`) fired on the right track. Free-text
noun phrase is fine; it does not need to match a fixed vocabulary.

### 2.6 `severity` — are we scoring magnitude or just presence?

3 levels: `subtle` / `clear` / `blatant`. Every score we produce is continuous, but we
have never checked whether that continuum tracks how bad the failure is. If our scores
separate blatant from clean but not subtle from clean, the useful operating point is
much narrower than our AUC suggests — and severity labels are the only way to see it.

### 2.7 `caption` and `generator` — needed *because* they are shortcuts

Both leak signal. Measured on the retrieval channel over 1,200 clips:

| Predictor | AUC |
|-----------|-----|
| DINOv2 k-NN, unrestricted | 0.699 |
| same k-NN, same-caption neighbours removed | 0.656 (**−0.043**) |
| same k-NN, restricted to same generator | 0.673 (−0.026) |
| generator identity alone, no video content | **0.527** |

So ~0.04 AUC of our retrieval result was caption identity, and the generator label by
itself beats chance. We need both fields recorded so we can *subtract* them. Without
them we cannot tell a physics detector from a "Sora clips are usually fine" detector.

This is also a **collection** requirement, not just an annotation one — see §4.

### 2.8 `rating2` — we currently have no ceiling

Every number we report is compared to chance (0.5). Nobody knows what a *human* scores
on this task. Double-annotate ~150 clips with a second person and we get:

* an inter-annotator agreement figure — the realistic ceiling, which is very unlikely
  to be 1.0;
* a read on which categories are ambiguous (if two humans disagree about whether a
  clip is a momentum or a collision failure, a model scoring 0.586 there may already
  be near the ceiling).

This is the cheapest credibility we can buy. 150 clips, one extra pass.

---

## 3. How many clips — the actual power calculation

Our per-category counts on VideoPhy-2's 704 annotated clips were badly unbalanced:

| Category | positives (of 704) |
|----------|--------------------|
| collision | 167 |
| deformation | 153 |
| momentum | 100 |
| gravity | 93 |
| fluid | 60 |
| friction | 35 |
| permanence | 21 |

95 % confidence interval on AUC (Hanley–McNeil, true AUC ≈ 0.70, 4 negatives per
positive):

| positives | 95 % CI on AUC | verdict |
|-----------|----------------|---------|
| 15 | ±0.160 | useless — cannot distinguish 0.70 from chance |
| 25 | ±0.124 | useless |
| 40 | ±0.098 | marginal |
| 60 | ±0.080 | minimum publishable |
| **80** | **±0.069** | **target** |
| 100 | ±0.062 | comfortable |
| 150 | ±0.050 | can compare two configs against each other |

**Our friction (0.707) and permanence (0.649) numbers have CIs of roughly ±0.13 and
±0.16 — neither is statistically distinguishable from chance.** They are in the report
as estimates, not findings. Please do not repeat that shape.

**Target: ≥80 positives per category.** With 7 categories, ~30 % clean clips, and
allowing multi-label overlap, that is roughly **900–1,100 clips total**. If the budget
is smaller, cut *categories*, not clips per category — 4 solid specialists beat 7
unmeasurable ones.

Also reserve the split before anyone looks at a number. `eval_run.py` writes
`split.json` on first run (25 % selection / 75 % held-out, stratified by rating) and
every downstream script reads it. This is not bureaucracy: cross-validated numbers on
the clips we had used to pick the configuration overstated our specialist AUCs by
**+0.140**. The split is the only thing that caught it.

---

## 4. Collection-side requirements (before annotation starts)

These cannot be fixed after the fact.

1. **Same caption, both outcomes.** Each prompt should appear with at least one good
   and one bad clip. Otherwise "caption → outcome" is learnable and worth ~0.04 AUC of
   fake performance (§2.7).
2. **Every generator produces both good and bad clips.** If one generator is only
   represented by its failures, generator identity becomes the label.
3. **Clean clips must be genuinely clean**, not "less bad". A clean clip with a subtle
   unlabelled violation poisons the negative pool for every specialist.
4. **Failure type must not correlate with scene type.** If every fluid violation is a
   kitchen and every gravity violation is a sports field, the specialists will learn
   scenes. Spread each category across several environments.
5. **Keep the source video.** We extract 8 frames at 512 px and freeze them, so all
   conditions see byte-identical pixels — but if we later need 16 frames or a
   different resolution, we need the originals.

---

## 5. File format

CSV or JSON. Column names are auto-detected from a list of aliases; anything unusual
can be mapped explicitly with `--map`.

```csv
id,rating,has_violation,category,t_start,t_end,rules,object,severity,caption,generator,rating2
clip_0001.mp4,2,1,gravity,1.20,1.90,"the mug hangs in the air after leaving the hand instead of falling",mug,blatant,"A person knocks a mug off a table",sora,2
clip_0002.mp4,5,0,,,,,,,"A ball rolls down a ramp and stops against a wall",wan,5
clip_0003.mp4,1,1,"collision;permanence",2.40,3.00,"the bat passes through the ball, then the ball disappears",baseball,clear,"A batter hits a baseball",cogvideo,1
```

Notes:

* `rating` — any numeric scale (1–5, 0–1, 0–100); it is rescaled to 1–5 internally.
  **Higher must mean more physically correct.** If your scale runs the other way, say
  so — do not silently invert it.
* `has_violation` — `1`/`0`, `true`/`false`, `yes`/`no` all parse. `is_clean` /
  `no_violation` are also accepted and inverted automatically.
* `category` — semicolon- or comma-separated for multi-label.
* `t_start` / `t_end` — seconds from clip start, float. Blank for clean clips; blank
  for a diffuse failure (set `severity` and leave the times empty).
* Blank cells are fine everywhere except `id` and `rating`.

Equivalent JSON is a list of objects with the same keys, or a dict keyed by id.

---

## 6. Pre-flight checklist

Run this before frames are extracted — it is fast, touches no video, and tells you
which tier you actually landed in:

```bash
python backend/scripts/eval_prepare.py --videos vids/ --labels labels.csv \
    --out data/ourbench --inspect
```

It prints the detected field mapping, how many rows matched a video file, the rating
range, the count of clips per category, an estimated CI per category, and a shortcut
audit (caption reuse, generator↔rating correlation). Everything it flags is cheaper to
fix during annotation than after.

Then, in order:

- [ ] every row's `id` matches a video file (0 unmatched)
- [ ] `rating` present and spans its full range
- [ ] ≥25 % of clips flagged clean
- [ ] ≥80 positives in every category we intend to report
- [ ] `other` category below ~10 % of failures
- [ ] `t_start`/`t_end` on ≥80 % of non-diffuse failures
- [ ] every caption appears with both a good and a bad clip
- [ ] every generator appears with both good and bad clips
- [ ] ~150 clips double-annotated for agreement
- [ ] `split.json` created and committed before anyone reads a score

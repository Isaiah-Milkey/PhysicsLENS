# Prompt Generation

Turns the initial frame of each physics video into a standardized JSON prompt
record, using a vision model to write the `scene` / `action` / `expected_outcome`
fields.

Backend is **ASU CreateAI**, using the same service token as the PhysicsLENS
project.

## Setup

```bash
pip install -r requirements.txt
```

Vendored into PhysicsLENS, so it reuses the repo's existing CreateAI token — no
setup needed. The key is read from `$CREATEAI_API_KEY`, then `--api-key`, then
the nearest `.env` searching upward from this directory, which finds
`PhysicsLENS/.env`. Every `.env` is gitignored; `.env.example` documents the
variable name for anyone running this folder standalone.

Note the source videos (`physicslens_robot_data/videos/`, 575 MB) are **not**
vendored — only `meta/` and `logs/`. `meta/index.csv` is all
`generate_testset_csv.py` needs; the videos are re-pullable from
`meta/seed100_manifest.json`.

## Frame naming

The script takes the taxonomy code and the observability flag **from the
filename**, so nothing has to be configured per frame:

```
<CODE>_<ID>_<obs|unobs>.<ext>

frames/RC_001_obs.jpg       -> Rigid Contact & Collision, observable
frames/MG_012_unobs.png     -> Magnetism, unobservable
frames/SLG_004_unobs.jpeg   -> Solid-Liquid-Gas, unobservable
```

`obs` / `observable` and `unobs` / `unobservable` are both accepted, and the
tokens are case-insensitive. Frames that don't match the pattern, or that use an
unknown code, are reported on stderr and skipped rather than silently dropped.
Subdirectories under `frames/` are searched too, so you can organize by domain
(`frames/MG/MG_012_unobs.png`) if that's easier.

Taxonomy codes: `RC FR SS MI FD DM GP SL LG GS SLG BY TH MG CX` — see
`taxonomy.py`, which is the single place to edit if the table changes.

## Running

```bash
python generate_prompts.py --dry-run          # list frames + show one full prompt, no API calls
python generate_prompts.py                    # frames/ -> output/
python generate_prompts.py --only MG,BY       # just those domains
python generate_prompts.py --overwrite        # regenerate records that already exist
python generate_prompts.py --model gpt5_1     # a different CreateAI model
```

Without `--overwrite`, frames that already have a JSON file are left alone and
their existing record is folded into the manifest — so re-running is cheap and
never clobbers hand-edited fields. A re-run that generates nothing new makes no
API call at all and doesn't need a key.

Other flags: `--frames-dir`, `--out-dir`, `--manifest`, `--provider`,
`--base-url` (swap in the beta/poc environment), `--no-recursive`,
`--temperature` (`-1` omits the parameter entirely).

## Output

`output/<ID>_<observability>.json` per frame, plus `output/prompts.json`
containing every record, a `pending_manual_outcome` list and a
`needs_marker_review` list.

```json
{
  "schema_version": "1.0",
  "id": "MG_012",
  "frame": "MG_012_unobs.png",
  "taxonomy": {
    "code": "MG",
    "domain": "Magnetism",
    "matter_state": "Solid",
    "type": "Implicit"
  },
  "observability": "unobservable",
  "scene": "Two gray blocks of similar size rest on a flat surface ...",
  "action": "A magnet is brought toward the blocks.",
  "expected_outcome": null,
  "needs_manual_outcome": true,
  "objects": ["gray block", "gray block"],
  "physics_focus": "magnetic attraction",
  "annotation_note": "Hidden property is not inferable from the frame; ...",
  "marker_leaks": [],
  "generated_by": {
    "backend": "createai-vision",
    "model": "openai/gpt4o",
    "generated_at": "2026-08-13T..."
  }
}
```

## Observable vs. unobservable

**Observable** — the governing property has a visual correlate in the frame (a
visibly larger box, a visibly rough surface). The model fills in every field,
`expected_outcome` included, and is told to ground the outcome in a property
that is actually visible.

**Unobservable** — the governing property cannot be read off the frame (two
identical boxes, one heavier; two identical rods, one magnetic). Here:

- `expected_outcome` is **never requested from the model**, and `build_record`
  additionally forces it to `null` regardless of what comes back — the invariant
  holds even if the model volunteers an outcome anyway.
- The record carries `"needs_manual_outcome": true` and the id appears in
  `pending_manual_outcome` in the manifest.
- The property under test lives in `property_spec` and is injected into
  `video_prompt`. See **Unobservable ablations** below.

### Unobservable ablations (`ablations.json`)

The unobservable subset is an **ablation on top of an observable base record**:
same frame, same scene and action, with one physical property injected into the
prompt text. That is what tests prompt adherence — whether the video model
actually renders a tungsten ball as heavy.

Specs live in `ablations.json`:

```json
[
  {
    "id": "RC_001_MI_tungsten",
    "base": "RC_001",
    "code": "MI",
    "property_spec": "The ball is solid tungsten and weighs about 45 kilograms, although it looks exactly like an ordinary soccer ball.",
    "physics_focus": "hidden mass / momentum transfer",
    "expected_outcome": null,
    "scene": "optional — overrides the base scene for this variant only"
  }
]
```

Ablations are derived from base records already in `output/`, so **they cost no
API calls**. `scene` and `action` are copied verbatim from the base: the only
difference between base and ablation prompt is `property_spec`, which is what
makes the comparison clean. One base frame can carry any number of variants.

Each ablation record gets a `video_prompt` — scene + property_spec + action,
assembled in that order so the property is read as part of the setup before the
action begins. `expected_outcome` is **not** included; it is ground truth for
scoring the generated video, not an instruction to it.

**Contradiction check.** A base scene that already asserts the property under
test breaks the ablation — gpt4o volunteers material unprompted ("a soccer ball
made of leather"), which flatly contradicts "the ball is solid tungsten".
`find_prompt_conflicts()` flags these into `prompt_conflicts` on the record and
`needs_prompt_review` in the manifest, and the run prints a warning. Fix by
either rewording `property_spec` or setting a neutral `scene` on that variant
(`scene_overridden: true` records that you did).

### The tape marker

Xinyuan's tape marker is internal ground truth and must never reach the prompt
text. Two defences, because the first one alone was not enough:

1. A `CRITICAL` clause in the query (not just the system prompt) telling the
   model to ignore any tape/sticker/coloured patch, never list it under
   `objects`, and treat two objects differing only by such a mark as identical.
2. `find_marker_leaks()` scans every generated field for marker vocabulary. On a
   hit the script re-asks, quoting the offending phrase back. If it still leaks
   after the retries, the record keeps a non-empty `marker_leaks` list, the id
   goes into `needs_marker_review`, and the run prints a warning — so a leak is
   always visible rather than silent.

In testing, gpt4o described the marker on the first attempt ("one block has a
small yellow mark") and produced a clean, marker-free description on the retry.
Still worth spot-checking unobservable records.

### Filling outcomes in by hand

Edit `expected_outcome` in the per-frame JSON under `output/`, then re-run the
script (without `--overwrite`) to rebuild the manifest. The id drops off
`pending_manual_outcome` automatically. Leave `needs_manual_outcome: true` as
the record of how that field was authored.

```bash
python -c "import json;print(json.load(open('output/prompts.json'))['pending_manual_outcome'])"
```

## Testset CSV (`generate_testset_csv.py`)

Separate, simpler deliverable: one video-generation prompt per video id in Xin's
`physicslens_robot_data/meta/index.csv`. **No observability, no taxonomy domain,
no expected outcome** — the prompt is scene + action only.

```bash
python generate_testset_csv.py --limit 3     # try a few first
python generate_testset_csv.py               # all -> testset_prompts.csv
python generate_testset_csv.py --only-id 7   # one id
```

Frames are matched to index rows by the leading number in the filename
(`007_..._frame10_720p.jpg` -> id 7); every frame's stem was verified against its
row's `.mp4` stem. Results cache to `testset_cache/<id>.json`, so re-runs are free
and a failure mid-way loses nothing. `--workers` controls concurrency (default 4;
81 frames take ~80s).

Columns: `id, video_file, task, frame, scene, action, prompt, action_review, note`.
`prompt` is the assembled `Scene: ...\nAction: ...`; `scene` and `action` are also
separate columns so the format can be changed without regenerating.

### The two constraints on `action`

1. **No outcome.** The benchmark tests whether a video model can infer that
   wiping a wet surface with a towel makes it clean, so saying so in the prompt
   destroys the test. "The robot wipes the cloth across the table surface" is
   right; "The robot cleans the table" is not. Xin's `task` column is
   goal-phrased ("Wipe the water off the table with a rag"), so the task is
   passed to the model only to identify *which* motion, with worked
   goal→motion examples. `find_outcome_leaks()` then scans each action and
   re-asks on a hit, quoting the offending phrase back; anything surviving lands
   in the `action_review` column.
2. **The whole manipulation, not just the reach.** Most frames are frame 0,
   before the motion starts, so the model's first instinct is "the robot moves
   its arm toward the bottle" — which would generate a video of reaching in which
   the physics under test never happens. The prompt demands the complete
   manipulation instead. This cut approach-only actions from 22/81 to 3.

## Unobservable variants (`generate_unobservable_csv.py`)

A second, **unobservable** version of the prompt for selected testset frames ->
`testset_prompts_unobservable.csv`. The frame does not change. Relative to the
observable prompt:

* the **scene** is rewritten: the visual cue that reveals the physical property
  is removed ("a cup filled with red liquid" -> "a translucent plastic cup") and
  a hidden property is stated in words instead;
* the **action is copied byte-for-byte** from the observable record and is never
  regenerated, so the paired prompts differ only in the scene;
* ground-truth fields are added: `hidden_property`, `hidden_property_value`,
  `expected_outcome`, `failure_signature`.

`expected_outcome` and `failure_signature` are scoring ground truth and are kept
**out of** the prompt — including them would hand the model the answer.

```bash
python generate_unobservable_csv.py --only-id 10    # one first
python generate_unobservable_csv.py                 # all ids in the spec
```

Which ids get a variant lives in `unobservable_ids.json`, grouped by task type.
The group selects the default property category and the worked example shown to
the model:

| group | category | example value |
|---|---|---|
| pouring | `viscosity` | high viscosity, honey-like consistency |
| wiping | `surface_condition` | silicone-coated, repels liquid |
| pushing | `mass` | very heavy, filled with dense material |
| deformable | `elasticity` | filled with dense gel, resists compression |

`hidden_property` must be one of: `mass, viscosity, density, friction,
elasticity, material_composition, thermal_state, surface_condition`.

### Overrides

`overrides` in `unobservable_ids.json` lets you force a category or hand-write
any generated field for one id. Id 71 uses one: the model kept dropping
"transparent" from the scene because the property it picked (filled with dense
metal) would be visible through a clear box — a fair objection. The override
keeps the box transparent and *looking empty* while stating it weighs 4 kg,
which makes the mass genuinely unobservable rather than self-contradictory.

### What is checked

Every row is re-checked at write time (not trusted from cache, so older records
are not exempted from newer rules) and anything suspect lands in the `review`
column:

- **`hidden_property`** is one of the eight categories.
- **A visual cue actually disappeared** — `removed_cues` lists which words.
- **The property is stated in the scene**, not merely recorded in a field.
- **No leftover contradiction** — "watery" surviving a viscosity rewrite. A word
  that also appears in `hidden_property_value` is exempt, since "appears empty
  but weighs 4 kg" is a deliberate apparent/actual contrast.
- **The property diverges from default behaviour.** Saying a plastic bag is
  "easily compressed" is useless — an ordinary bag already does that, so a model
  ignoring the text produces the same video. Caught on id 29's first pass.
- **The scene was not over-rewritten** — `scene_similarity` (currently 0.67–0.90,
  median 0.85). Note this uses `autojunk=False`; the default `SequenceMatcher`
  heuristic reports ~0.03 for what is really a one-clause edit on long strings.
- **The action still resolves.** The rewrite must not strip a descriptor the
  fixed action uses to name its target — 5 of 30 first-pass rows said "a small
  container" while the action said "the transparent box".

Fixable faults are re-asked automatically, quoting the problem back.

## CreateAI API notes

Worth writing down, since these cost some trial and error:

- Vision is **not** a `/vision` path. It is `POST {base}/query` with
  `"endpoint": "vision"` in the body. Base URL `https://api-main.aiml.asu.edu`
  (prod; `-beta` and `-poc` also exist).
- `image_file` must be a **data URI** (`data:image/png;base64,...`). Raw base64
  is rejected with *"Expected a valid URL"*.
- `request_source` must be `"override_params"` for a project service token.
- The OpenAI-compatible route (`/v1/chat/completions`) **accepts `image_url`
  content blocks, silently ignores them, and answers from imagination** — it
  returned "a red apple on a wooden table" for a picture of a ball on a ramp.
  Do not use it for anything involving images.
- `response_format` supports only `{"type": "json"}` — there is no
  `json_schema`/strict mode, so replies are validated and repaired in
  `parse_model_json()`. Without a system prompt forbidding it, replies often
  arrive wrapped in ```` ```json ```` fences.
- `GET {base}/v1/models` lists the available models (123 at time of writing);
  vision-capable OpenAI ones include `openai/gpt4o`, `openai/gpt4_1`,
  `openai/gpt5_1`.
- Docs: <https://docs.aiml.asu.edu/endpoints/vision>

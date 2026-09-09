#!/usr/bin/env python3
"""Generate standardized physics-reasoning prompts from initial video frames.

For each frame image the script:
  1. reads the taxonomy code and the observable/unobservable flag from the filename
     (see taxonomy.py -- e.g. ``RC_001_obs.jpg``, ``MG_012_unobs.png``),
  2. asks a vision model to describe the scene and the action about to happen,
  3. writes one JSON file per frame plus a combined manifest.

Observability handling
----------------------
observable   : the model fills scene, action AND expected_outcome.
unobservable : the model fills scene and action only. ``expected_outcome`` is
               forced to null and flagged for manual completion, because the
               governing property (hidden mass, magnetism, ...) is not visible
               in the frame. The tape marker in the photo is internal ground
               truth only -- the model is explicitly told to ignore it so it
               never leaks into the prompt text.

Backend: ASU CreateAI vision
----------------------------
Calls the CreateAI **vision** endpoint, which is POST {base}/query with
``"endpoint": "vision"`` in the body -- NOT a /vision path, and NOT the
OpenAI-compatible /v1 route (that one accepts image_url blocks, silently drops
them, and answers from imagination -- verified 2026-08-13).

Key facts this client depends on:
  * ``image_file`` must be a data URI (``data:image/png;base64,...``); raw
    base64 is rejected with "Expected a valid URL".
  * ``request_source`` must be ``"override_params"`` for a project service token.
  * ``response_format {"type": "json"}`` is the only structured mode -- there is
    no json_schema/strict mode, so replies are validated and repaired here.

Usage
-----
    export CREATEAI_API_KEY=...        # or put it in .env next to this script
    python generate_prompts.py --frames-dir frames --out-dir output
    python generate_prompts.py --dry-run          # no API calls, prints the plan
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional

from taxonomy import (
    FOCUS_HINTS,
    TAXONOMY,
    OBSERVABLE,
    UNOBSERVABLE,
    FilenameError,
    FrameSpec,
    find_frames,
    parse_frame_filename,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# The key is read from $CREATEAI_API_KEY, then --api-key, then a .env file next
# to this script (same convention as the PhysicsLENS project). Never hard-code it.
API_KEY_ENV = "CREATEAI_API_KEY"

# CreateAI environments: prod / beta / poc.
DEFAULT_BASE_URL = "https://api-main.aiml.asu.edu"

# provider/model as listed by GET {base}/v1/models. gpt4o is a solid, cheap
# vision default; gpt5_1 also works and is more literal about spatial layout.
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt4o"

DEFAULT_FRAMES_DIR = "frames"
DEFAULT_ABLATIONS = "ablations.json"
DEFAULT_OUT_DIR = "output"
DEFAULT_MANIFEST = "prompts.json"

SCHEMA_VERSION = "1.0"

MAX_RETRIES = 3
RETRY_BACKOFF_S = 4.0
REQUEST_TIMEOUT_S = 120

# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """\
You are annotating the FIRST FRAME of a short real-world physics video for a \
physical-reasoning benchmark. The frame shows a scene an instant before a \
physical event unfolds.

Write in plain, neutral, third-person English. Be concrete and specific about \
objects, materials, spatial relations and geometry. Never mention the camera, \
the image, the photo, the frame, or the annotation task itself: describe the \
scene as a physical situation, not as a picture.

Ignore any small piece of tape, sticker, dot or other tiny marker attached to \
an object. Those markers are internal bookkeeping for the dataset authors. Do \
not describe them, do not refer to them, and do not treat them as physically \
meaningful.

Return only the JSON object requested by the schema."""

FIELD_GUIDE_COMMON = """\
Fields:

- scene: 2-4 sentences. The static setup at the moment of the frame. Objects \
present, their materials and approximate sizes, how they are arranged, what \
supports what, and any relevant surface or container. State only what is \
visible; do not speculate about hidden properties.

- action: 1-2 sentences, present or immediate-future tense. The single physical \
action that is about to occur or is just beginning (e.g. "The hand releases \
the block", "The pitcher is tipped over the glass"). Describe the action only, \
not its result.

- objects: the salient physical objects in the scene, as short noun phrases.

- physics_focus: one short phrase naming the physical principle the scenario \
tests."""

FIELD_GUIDE_OBSERVABLE = """\
- expected_outcome: 2-4 sentences. What physically happens after the action, \
and why. Ground every claim in a property that is visible in the frame (size, \
shape, material, texture, incline, fill level, contact geometry). Describe the \
outcome deterministically, as the single most likely real-world result. Do not \
hedge with "might" or "could", and do not offer alternatives."""

USER_TEMPLATE = """\
Physics domain: {code} - {domain}
Matter state: {matter_state}
Scenario type: {type}
Pay particular attention to: {focus}

{field_guide}

{observability_note}"""

NOTE_OBSERVABLE = """\
This scenario is OBSERVABLE: the property that governs the outcome is visible \
in the frame. Fill in every field, including expected_outcome."""

NOTE_UNOBSERVABLE = """\
This scenario is UNOBSERVABLE: the property that governs the outcome (for \
example a hidden mass difference, or which of two identical objects is \
magnetic) CANNOT be determined from this frame. Do not guess it and do not \
hint at it. Describe only the visible setup and the action. The outcome will \
be supplied separately by a human annotator, so no outcome field is requested \
of you."""


CONTRACT_TEMPLATE = """\
Return a single JSON object with exactly these keys: {keys}.
No markdown fences, no commentary, nothing outside the JSON object."""

# Repeated in the query, not just the system prompt: with the instruction only in
# model_params.system_prompt the model still described the tape ("one block has a
# small yellow rectangle attached") and even listed it under objects.
MARKER_CLAUSE = """\
CRITICAL: one object may carry a small piece of tape, a sticker, a coloured \
patch or a similar small mark. That is a private annotation marker, not part of \
the experiment. Describe the scene as if it were not there: never mention it, \
never use it to distinguish or identify an object, and never list it under \
objects. Two objects that differ only by such a mark must be described as \
identical."""

# Post-hoc leak check. The prompt is the primary defence; this catches the rest.
_MARKER_PATTERNS = [
    r"\b(tape|sticker|decal|adhesive|post-?it|sticky note)\b",
    r"\b(?:small|tiny|little)?\s*(?:yellow|orange|red|green|blue|white|black)\s+"
    r"(rectangle|square|patch|strip|dot|spot|mark|tab|label)\b",
    r"\b(marker|label)\s+(?:attached|affixed|stuck|placed)\b",
]
_MARKER_RE = re.compile("|".join(_MARKER_PATTERNS), re.IGNORECASE)


def find_marker_leaks(payload: dict[str, Any]) -> list[str]:
    """Return the offending snippets if the reply describes the tape marker."""
    leaks: list[str] = []
    for field in ("scene", "action", "expected_outcome", "objects", "physics_focus"):
        value = payload.get(field)
        if value is None:
            continue
        text = " ; ".join(str(v) for v in value) if isinstance(value, list) else str(value)
        for m in _MARKER_RE.finditer(text):
            leaks.append(f"{field}: {m.group(0).strip()}")
    return leaks


def expected_keys(unobservable: bool) -> list[str]:
    """Field list for a frame. ``expected_outcome`` is simply never requested for
    unobservable frames, so the model is not even invited to guess it."""
    keys = ["scene", "action"]
    if not unobservable:
        keys.append("expected_outcome")
    return keys + ["objects", "physics_focus"]


def build_user_prompt(spec: FrameSpec) -> str:
    d = spec.domain
    unobs = spec.is_unobservable
    guide = FIELD_GUIDE_COMMON if unobs else f"{FIELD_GUIDE_COMMON}\n\n{FIELD_GUIDE_OBSERVABLE}"
    keys = ", ".join(f'"{k}"' for k in expected_keys(unobs))
    return USER_TEMPLATE.format(
        code=d.code,
        domain=d.domain,
        matter_state=d.matter_state,
        type=d.type,
        focus=FOCUS_HINTS.get(d.code, "the dominant physical interaction"),
        field_guide=guide,
        observability_note=NOTE_UNOBSERVABLE if unobs else NOTE_OBSERVABLE,
    ) + "\n\n" + MARKER_CLAUSE + "\n\n" + CONTRACT_TEMPLATE.format(keys=keys)


# --------------------------------------------------------------------------- #
# Model call
# --------------------------------------------------------------------------- #

def encode_image(path: str) -> str:
    """CreateAI requires a data URI here; raw base64 is rejected outright."""
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def load_dotenv(path: str) -> None:
    """Minimal KEY=VALUE reader. Does not overwrite a variable already exported."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def resolve_api_key(cli_key: Optional[str]) -> str:
    if cli_key:
        return cli_key
    # Walk up from the script directory so a .env at the repo root is found too
    # -- this module is vendored into PhysicsLENS/prompt_generation/, whose key
    # lives one level up in PhysicsLENS/.env. Nearest .env wins (load_dotenv
    # uses setdefault, and we ascend).
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        load_dotenv(os.path.join(here, ".env"))
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        sys.exit(
            f"No API key found. Set {API_KEY_ENV} in the environment, pass "
            f"--api-key, or put {API_KEY_ENV}=... in a .env file next to this script."
        )
    return key


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_model_json(raw: str, keys: list[str]) -> dict[str, Any]:
    """Parse the model's reply into a dict with every key in ``keys``.

    CreateAI's response_format only offers {"type": "json"} -- there is no strict
    schema mode -- and in practice replies still arrive wrapped in ```json fences
    often enough to matter. So: strip fences, fall back to the outermost {...},
    then verify the contract ourselves.
    """
    text = _FENCE_RE.sub("", (raw or "").strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError(f"reply was not JSON: {text[:200]!r}")
        data = json.loads(m.group(0))

    if not isinstance(data, dict):
        raise ValueError(f"reply was not a JSON object: {text[:200]!r}")

    missing = [k for k in keys if k not in data or data[k] in (None, "", [])]
    if missing:
        raise ValueError(f"reply missing/empty field(s): {', '.join(missing)}")
    return data


def call_model(
    api_key: str,
    base_url: str,
    provider: str,
    model: str,
    spec: FrameSpec,
    temperature: Optional[float],
) -> dict[str, Any]:
    """One frame -> one validated JSON payload, via the CreateAI vision endpoint.

    Retries transport errors AND contract violations, since a reply that omits a
    field is usually fixed by simply asking again.
    """
    try:
        import requests
    except ImportError:
        sys.exit("The requests package is required: pip install -r requirements.txt")

    keys = expected_keys(spec.is_unobservable)
    model_params: dict[str, Any] = {"system_prompt": SYSTEM_PROMPT}
    if temperature is not None:
        model_params["temperature"] = temperature

    base_query = build_user_prompt(spec)
    image = encode_image(spec.path)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{base_url.rstrip('/')}/query"

    query = base_query
    best: Optional[dict[str, Any]] = None   # last valid reply, even if it leaked
    last_err: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        payload = {
            "endpoint": "vision",
            # Required for a project service token; harmless for developer tokens.
            "request_source": "override_params",
            "query": query,
            "image_file": image,
            "model_provider": provider,
            "model_name": model,
            "model_params": model_params,
            "response_format": {"type": "json"},
            "enable_search": False,
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_S)
            if not resp.ok:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            body = resp.json()
            raw = body.get("response")
            if raw is None:
                raise RuntimeError(f"no 'response' field in reply: {str(body)[:300]}")
            data = parse_model_json(raw if isinstance(raw, str) else json.dumps(raw), keys)
        except Exception as err:  # noqa: BLE001 - transport, HTTP and contract errors alike
            last_err = err
            if attempt == MAX_RETRIES:
                break
            wait = RETRY_BACKOFF_S * attempt
            print(f"    attempt {attempt} failed ({err}); retrying in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            continue

        best = data
        leaks = find_marker_leaks(data)
        if not leaks:
            return data
        if attempt == MAX_RETRIES:
            break
        # Re-ask with the offending phrases quoted back; a generic reminder alone
        # tends not to shift it.
        print(f"    marker leak ({'; '.join(leaks)}); re-asking", file=sys.stderr)
        query = (
            f"{base_query}\n\nYour previous answer violated the marker rule: it "
            f"mentioned {'; '.join(repr(l.split(': ', 1)[-1]) for l in leaks)}. "
            "Rewrite it with no reference whatsoever to any tape, sticker, patch "
            "or coloured mark, and do not use it to tell objects apart."
        )

    if best is not None:
        return best
    raise RuntimeError(f"model call failed after {MAX_RETRIES} attempts: {last_err}")


# --------------------------------------------------------------------------- #
# Record assembly
# --------------------------------------------------------------------------- #

def build_record(spec: FrameSpec, payload: dict[str, Any], model: str, frames_dir: str) -> dict[str, Any]:
    unobs = spec.is_unobservable
    return {
        "schema_version": SCHEMA_VERSION,
        "id": spec.sample_id,
        "frame": os.path.relpath(spec.path, frames_dir),
        "taxonomy": spec.domain.as_dict(),
        "observability": spec.observability,
        "scene": str(payload["scene"]).strip(),
        "action": str(payload["action"]).strip(),
        # Hard invariant: an unobservable frame never carries a model-written
        # outcome, whatever the model returned.
        "expected_outcome": None if unobs else str(payload["expected_outcome"]).strip(),
        "video_prompt": build_video_prompt(
            str(payload["scene"]), str(payload["action"])),
        "needs_manual_outcome": unobs,
        # Without a strict schema the model occasionally returns objects as dicts
        # or as one comma-joined string, so normalise rather than trust.
        "objects": _as_str_list(payload.get("objects")),
        "physics_focus": str(payload.get("physics_focus", "")).strip(),
        "annotation_note": (
            "Hidden property is not inferable from the frame; fill expected_outcome "
            "by hand from the setup ground truth. The tape marker identifies the "
            "special object and must not appear in any prompt text."
            if unobs else None
        ),
        # Non-empty only if the model still described the marker after retries;
        # such a record needs the wording fixed by hand before use.
        "marker_leaks": find_marker_leaks(payload),
        "generated_by": {
            "backend": "createai-vision",
            "model": model,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    if isinstance(value, list):
        out = []
        for v in value:
            if isinstance(v, dict):  # e.g. {"name": "red ball", ...}
                v = v.get("name") or v.get("object") or next(iter(v.values()), "")
            s = str(v).strip()
            if s:
                out.append(s)
        return out
    return [str(value).strip()]


# An ablation only isolates the injected property if the base scene does not
# already assert a competing value for it. gpt4o volunteers material ("made of
# synthetic material") and size unprompted, which directly contradicts a
# property_spec like "the ball is solid tungsten".
_MATERIAL_RE = re.compile(
    r"\b(?:made (?:out )?of|moulded from|molded from|constructed from|"
    r"consists? of)\s+((?:[a-z][a-z\-]*)(?:\s+(?!and\b|which\b|that\b|is\b|"
    r"stands?\b|appears?\b|rests?\b|sits?\b|lies?\b)[a-z][a-z\-]*){0,2})",
    re.IGNORECASE)
_MASS_RE = re.compile(
    r"\b(weigh(?:s|ing|t)?|mass of|kilograms?|kg\b|grams?\b|pounds?\b|lbs?\b)",
    re.IGNORECASE)
_FRICTION_RE = re.compile(r"\b(smooth|rough|slippery|frictionless|grippy|polished)\b",
                          re.IGNORECASE)


def find_prompt_conflicts(scene: str, property_spec: str) -> list[str]:
    """Flag base-scene claims that fight the injected property."""
    conflicts: list[str] = []
    prop = property_spec.lower()

    targets_material = bool(_MATERIAL_RE.search(prop)) or any(
        w in prop for w in ("tungsten", "steel", "foam", "rubber", "wood",
                            "plastic", "lead", "aluminium", "aluminum", "ice"))
    targets_mass = bool(_MASS_RE.search(prop))
    targets_friction = bool(_FRICTION_RE.search(prop))

    if targets_material:
        for m in _MATERIAL_RE.finditer(scene):
            note = f"scene states material {m.group(1).strip().rstrip(',.')!r}"
            if note not in conflicts:
                conflicts.append(note)
    if targets_mass and _MASS_RE.search(scene):
        conflicts.append("scene already states a mass/weight")
    if targets_friction and _FRICTION_RE.search(scene):
        conflicts.append("scene already characterises the surface texture")
    return conflicts


def build_video_prompt(scene: str, action: str, property_spec: Optional[str] = None) -> str:
    """The text actually handed to the video-generation model.

    expected_outcome is deliberately NOT included -- that is ground truth for
    scoring the generated video, not an instruction to it. For an ablation the
    property statement sits between scene and action, so the model reads the
    hidden property as part of the setup before the action begins.
    """
    parts = [scene.strip()]
    if property_spec:
        parts.append(property_spec.strip())
    parts.append(action.strip())
    return " ".join(p for p in parts if p)


def load_ablations(path: str) -> list[dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):           # allow {"ablations": [...]}
        data = data.get("ablations", [])
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of ablation specs")
    return data


def build_ablation_record(spec: dict[str, Any], base: dict[str, Any],
                          frames_dir: str) -> dict[str, Any]:
    """Derive an unobservable ablation record from an observable base record.

    scene/action are copied VERBATIM from the base, never regenerated: the whole
    point of the ablation is that the only thing that changed is the injected
    property. Regenerating (or re-describing) would introduce wording drift and
    confound the comparison -- and it would cost an extra API call per variant
    for a description we already have.
    """
    code = str(spec.get("code", base["taxonomy"]["code"])).upper()
    if code not in TAXONOMY:
        raise ValueError(f"unknown taxonomy code {code!r}")
    prop = spec.get("property_spec")
    if not prop:
        raise ValueError("missing 'property_spec' (the text stating the hidden property)")

    # A base scene that asserts the very property under test (gpt4o likes to say
    # "made of leather") contradicts property_spec. Override it here rather than
    # editing the base record, which other ablations still depend on.
    scene = spec.get("scene", base["scene"])
    action = spec.get("action", base["action"])
    frame = spec.get("frame", base["frame"])   # usually the base frame, reused

    return {
        "schema_version": SCHEMA_VERSION,
        "id": spec["id"],
        "frame": frame,
        "taxonomy": TAXONOMY[code].as_dict(),
        "observability": UNOBSERVABLE,
        "base_id": base["id"],
        "property_spec": prop.strip(),
        "scene": scene,
        "action": action,
        "video_prompt": build_video_prompt(scene, action, prop),
        "expected_outcome": spec.get("expected_outcome"),
        "needs_manual_outcome": spec.get("expected_outcome") in (None, ""),
        "objects": list(base.get("objects", [])),
        "physics_focus": spec.get("physics_focus", base.get("physics_focus", "")),
        "annotation_note": (
            "Ablation of "
            + base["id"]
            + ": scene/action copied verbatim from the base record, only "
            "property_spec differs. Fill expected_outcome by hand -- it is the "
            "ground truth for whether the generated video obeyed property_spec."
        ),
        "marker_leaks": base.get("marker_leaks", []),
        "prompt_conflicts": find_prompt_conflicts(scene, prop),
        "scene_overridden": "scene" in spec,
        "generated_by": dict(base.get("generated_by", {}), derived_from=base["id"]),
    }


def out_path(out_dir: str, spec: FrameSpec) -> str:
    return os.path.join(out_dir, f"{spec.sample_id}_{spec.observability}.json")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames-dir", default=DEFAULT_FRAMES_DIR, help="directory of initial frames")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="directory for per-frame JSON")
    p.add_argument("--manifest", default=DEFAULT_MANIFEST, help="combined JSON written into --out-dir")
    p.add_argument("--ablations", default=DEFAULT_ABLATIONS,
                   help="JSON list of unobservable ablation specs derived from base records")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"CreateAI model name, e.g. gpt4o / gpt5_1 (default: {DEFAULT_MODEL})")
    p.add_argument("--provider", default=DEFAULT_PROVIDER,
                   help=f"CreateAI model provider (default: {DEFAULT_PROVIDER})")
    p.add_argument("--api-key", default=None, help=f"overrides ${API_KEY_ENV}")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help=f"CreateAI environment base URL (default: {DEFAULT_BASE_URL})")
    p.add_argument("--temperature", type=float, default=0.2,
                   help="sampling temperature; use --temperature -1 to omit it entirely")
    p.add_argument("--only", default=None,
                   help="comma-separated taxonomy codes to process, e.g. RC,BY,MG")
    p.add_argument("--overwrite", action="store_true", help="regenerate frames that already have JSON")
    p.add_argument("--dry-run", action="store_true", help="list what would be done, make no API calls")
    p.add_argument("--no-recursive", action="store_true", help="do not descend into subdirectories")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    frames_dir = os.path.abspath(args.frames_dir)
    out_dir = os.path.abspath(args.out_dir)

    if not os.path.isdir(frames_dir):
        sys.exit(f"frames directory not found: {frames_dir}")

    files = find_frames(frames_dir, recursive=not args.no_recursive)
    if not files:
        sys.exit(f"no images found in {frames_dir}")

    only = {c.strip().upper() for c in args.only.split(",")} if args.only else None

    specs: list[FrameSpec] = []
    skipped: list[str] = []
    for f in files:
        try:
            spec = parse_frame_filename(f)
        except FilenameError as err:
            skipped.append(str(err))
            continue
        if only and spec.domain.code not in only:
            continue
        specs.append(spec)

    for msg in skipped:
        print(f"SKIP  {msg}", file=sys.stderr)

    if not specs:
        sys.exit("no frames matched the naming convention (and --only filter)")

    n_unobs = sum(s.is_unobservable for s in specs)
    print(f"{len(specs)} frame(s): {len(specs) - n_unobs} observable, {n_unobs} unobservable")

    if args.dry_run:
        for s in specs:
            print(f"  {s.sample_id:10s} {s.observability:12s} {s.domain.domain:32s} {os.path.relpath(s.path, frames_dir)}")
        print("\n--- example prompt ---")
        print(build_user_prompt(specs[0]))
        return 0

    temperature = None if args.temperature is not None and args.temperature < 0 else args.temperature

    # Resolved on first real use, so re-running purely to rebuild the manifest
    # (e.g. after hand-filling outcomes) needs neither key nor network.
    api_key: Optional[str] = None
    def get_key() -> str:
        nonlocal api_key
        if api_key is None:
            api_key = resolve_api_key(args.api_key)
        return api_key

    records: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []

    for i, spec in enumerate(specs, 1):
        dest = out_path(out_dir, spec)
        if os.path.exists(dest) and not args.overwrite:
            print(f"[{i}/{len(specs)}] {spec.sample_id}: exists, skipping (use --overwrite)")
            with open(dest, encoding="utf-8") as fh:
                records.append(json.load(fh))
            continue

        print(f"[{i}/{len(specs)}] {spec.sample_id} ({spec.observability}) ...", flush=True)
        try:
            payload = call_model(get_key(), args.base_url, args.provider,
                                 args.model, spec, temperature)
            record = build_record(spec, payload, f"{args.provider}/{args.model}", frames_dir)
        except Exception as err:  # noqa: BLE001
            print(f"    FAILED: {err}", file=sys.stderr)
            failures.append((spec.sample_id, str(err)))
            continue

        write_json(dest, record)
        records.append(record)

    # --- Unobservable ablations -------------------------------------------- #
    # Derived from the base records above, so they cost no API calls: only the
    # injected property_spec differs from the base prompt.
    abl_path = args.ablations if os.path.isabs(args.ablations) else os.path.abspath(args.ablations)
    by_id = {r["id"]: r for r in records}
    for spec in load_ablations(abl_path):
        try:
            base = by_id.get(spec.get("base"))
            if base is None:
                raise ValueError(f"base record {spec.get('base')!r} not found "
                                 "(generate the base frame first)")
            rec = build_ablation_record(spec, base, frames_dir)
        except Exception as err:  # noqa: BLE001
            print(f"ABLATION SKIP {spec.get('id', '?')}: {err}", file=sys.stderr)
            failures.append((str(spec.get("id", "?")), str(err)))
            continue
        dest = os.path.join(out_dir, f"{rec['id']}_{rec['observability']}.json")
        # An ablation's expected_outcome is hand-written by definition -- there is
        # nothing to regenerate -- so recover it even under --overwrite. Losing a
        # human label to a routine re-run is not an acceptable failure mode.
        # Precedence: ablations.json (explicit) > existing file > null.
        if rec["expected_outcome"] in (None, "") and os.path.exists(dest):
            with open(dest, encoding="utf-8") as fh:
                existing = json.load(fh)
            if existing.get("expected_outcome") not in (None, ""):
                rec["expected_outcome"] = existing["expected_outcome"]
                rec["needs_manual_outcome"] = False
                rec["outcome_source"] = "recovered from output/"
        write_json(dest, rec)
        records.append(rec)
        print(f"  ablation {rec['id']} <- {base['id']}")

    records.sort(key=lambda r: r["id"])
    manifest_path = os.path.join(out_dir, args.manifest)
    leaked = [r["id"] for r in records if r.get("marker_leaks")]
    conflicted = [r["id"] for r in records if r.get("prompt_conflicts")]
    write_json(manifest_path, {
        "schema_version": SCHEMA_VERSION,
        "count": len(records),
        "pending_manual_outcome": [r["id"] for r in records if r["needs_manual_outcome"] and r["expected_outcome"] is None],
        "needs_marker_review": leaked,
        "needs_prompt_review": conflicted,
        "records": records,
    })

    pending = sum(1 for r in records if r["needs_manual_outcome"] and r["expected_outcome"] is None)
    print(f"\nwrote {len(records)} record(s) -> {manifest_path}")
    if pending:
        print(f"{pending} unobservable record(s) need expected_outcome filled in by hand.")
    if conflicted:
        print(f"WARNING: {len(conflicted)} ablation(s) contradict their base scene; "
              f"reword the base scene or property_spec: {', '.join(conflicted)}",
              file=sys.stderr)
    if leaked:
        print(f"WARNING: {len(leaked)} record(s) still mention the marker and need "
              f"manual rewording: {', '.join(leaked)}", file=sys.stderr)
    if failures:
        print(f"{len(failures)} frame(s) failed:", file=sys.stderr)
        for sid, err in failures:
            print(f"  {sid}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

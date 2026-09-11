#!/usr/bin/env python3
"""Second, UNOBSERVABLE version of the prompt for selected testset frames.

The frame does not change. Relative to the observable prompt:

  * the SCENE is rewritten -- the visual cue that reveals the physical property
    is removed ("a cup filled with red liquid" -> "a translucent plastic cup"),
    and a hidden property is stated in plain language instead;
  * the ACTION is copied word-for-word from the observable record. It is never
    regenerated, so the only difference between the paired prompts is the scene;
  * ground-truth fields are added: hidden_property, hidden_property_value,
    expected_outcome and failure_signature.

expected_outcome and failure_signature are scoring ground truth and are kept OUT
of the prompt -- putting them in would hand the video model the answer.

Which ids get a variant, and their task group, lives in `unobservable_ids.json`.

Usage
-----
    python generate_unobservable_csv.py --only-id 10     # try one first
    python generate_unobservable_csv.py                  # all 30
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Optional

from generate_prompts import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    MAX_RETRIES,
    RETRY_BACKOFF_S,
    REQUEST_TIMEOUT_S,
    encode_image,
    parse_model_json,
    resolve_api_key,
)
from generate_testset_csv import DEFAULT_FRAMES, build_prompt

DEFAULT_SPEC = "unobservable_ids.json"
DEFAULT_OBSERVABLE = "testset_prompts.csv"
DEFAULT_OUT = "testset_prompts_unobservable.csv"
DEFAULT_CACHE = "unobservable_cache"

# The fixed vocabulary for hidden_property. A value outside this list is a
# contract violation and is retried.
PROPERTY_CATEGORIES = [
    "mass", "viscosity", "density", "friction", "elasticity",
    "material_composition", "thermal_state", "surface_condition",
]

# Per task group: the category to prefer and a worked example, phrased exactly
# the way the value and the scene sentence should read.
GROUP_GUIDE: dict[str, dict[str, str]] = {
    "pouring": {
        "category": "viscosity",
        "alternatives": "density, thermal_state",
        "value": "high viscosity, honey-like consistency",
        "scene_says": "the liquid is thick and viscous, similar to honey",
    },
    "wiping": {
        "category": "surface_condition",
        "alternatives": "friction, material_composition",
        "value": "silicone-coated, repels liquid",
        "scene_says": "the surface has been coated with silicone, causing liquid "
                      "to bead up and resist absorption",
    },
    "pushing": {
        "category": "mass",
        "alternatives": "friction, density",
        "value": "very heavy, filled with dense material",
        "scene_says": "the object contains 3kg of hidden weights and will strongly "
                      "resist being pushed",
    },
    "deformable": {
        "category": "elasticity",
        "alternatives": "material_composition, mass",
        "value": "filled with dense gel, resists compression",
        "scene_says": "the ball is filled with dense gel rather than air and "
                      "strongly resists compression",
    },
}

SYSTEM_PROMPT = """\
You are building the UNOBSERVABLE half of a paired physical-reasoning benchmark \
for video-generation models.

Each pair shares one frame. The observable prompt describes what is visible. The \
unobservable prompt describes the same frame with the give-away visual cue \
removed and a hidden physical property stated in words instead. The benchmark \
then measures whether a generated video obeys a property it can only have learnt \
from the text.

Write plain, neutral, third-person English. Never mention the camera, the image, \
the photo, the frame, or the annotation task. Return only the requested JSON \
object -- no markdown fences, no commentary."""

QUERY_TEMPLATE = """\
Here is the OBSERVABLE scene description already written for this frame:

  "{observable_scene}"

The action, which you must NOT rewrite and must NOT refer to, is:

  "{action}"

Task group: {group}. Logged task: "{task}"

Produce a JSON object with exactly these keys: "scene", "hidden_property", \
"hidden_property_value", "expected_outcome", "failure_signature".

- scene: the observable scene REWRITTEN. Two changes, nothing else:
    (a) REMOVE the visual cue that reveals the physical property -- the colour, \
texture, fill-level or material wording that lets a viewer guess it. Remove ONLY \
that cue. Every other descriptor stays, even colours and materials: if the \
property is the table's coating, the cloth is still "yellow" and the bowl is \
still "ceramic". Keep every other object and spatial relation intact, in the same \
order, with the same wording wherever the cue is not involved.
    NEVER remove a word the action uses to refer to an object. The action above \
is fixed and will follow your scene verbatim -- if it says "the transparent box" \
or "the pink towel", your scene must still identify that object the same way, or \
the reader cannot tell which object is meant.
    (b) ADD one clause stating the hidden property in plain language, phrased so \
it is clearly a fact about the object, e.g. "{scene_says}".
  Do not describe the action or its result in the scene.

- hidden_property: exactly one of: {categories}
  For this task group prefer "{category}" (alternatives if the frame suits them \
better: {alternatives}).

- hidden_property_value: a short specific description of the property, e.g. \
"{value}".
  It MUST be extreme enough that a video obeying it looks VISIBLY different from \
one ignoring it. If a viewer could not tell the two apart, make it more extreme.
  CRITICAL: the property must make the outcome DIVERGE from how this object would \
ordinarily behave -- normally by resisting or impeding the action. Never choose a \
property whose result matches the default: saying a plastic bag is "easily \
compressed" is useless, because an ordinary bag already compresses easily and a \
model ignoring the text would produce the same video. Say it resists instead.

- expected_outcome: what correct physics looks like, in visual falsifiable terms \
-- something a person could check by watching the clip and say yes or no to. \
Describe what is seen, not the principle behind it.

- failure_signature: what the clip looks like if the model IGNORES the stated \
property and defaults to the ordinary version of this scene.

Worked example, for a syringe drawing from a cup:
  observable scene: "...a translucent plastic cup filled with red liquid..."
  scene: "...a translucent plastic cup. The cup contains a highly viscous fluid \
similar in consistency to honey, which will strongly resist being drawn into the \
syringe."
  hidden_property: "viscosity"
  hidden_property_value: "high viscosity, honey-like consistency"
  expected_outcome: "The syringe plunger moves back very slowly and fluid rises \
incompletely into the barrel."
  failure_signature: "Fluid draws into syringe as freely as water despite stated \
high viscosity."
"""

# Wording that would contradict the stated property if it survived the rewrite.
_CONTRADICTIONS: dict[str, list[str]] = {
    "viscosity": [r"\bwatery\b", r"\bthin,? (?:clear )?liquid\b", r"\brunny\b",
                  r"\bfree[- ]flowing\b"],
    "mass": [r"\blight(?:weight)?\b", r"\bempty\b", r"\bhollow\b"],
    "density": [r"\blight(?:weight)?\b", r"\bhollow\b"],
    "elasticity": [r"\bsquishy\b", r"\bair[- ]filled\b", r"\bsoft(?:,| and) (?:squishy|pliable)\b"],
    "friction": [r"\bsmooth and slippery\b"],
    "surface_condition": [r"\babsorbent\b", r"\bporous\b", r"\bsoaks? up\b"],
    "material_composition": [],
    "thermal_state": [],
}

# Cue vocabulary: words whose disappearance from the scene shows a visual cue was
# actually removed. Used for reporting, not enforcement.
_CUE_WORDS = re.compile(
    r"\b(red|blue|green|yellow|orange|purple|pink|brown|white|black|clear|"
    r"silver|gold|golden|grey|gray|beige|tan|cream|ivory|turquoise|violet|"
    r"translucent|transparent|amber|milky|dark|pale|colou?red|"
    r"metal|metallic|plastic|rubber|wooden|wood|glass|foam|cloth|fabric|paper|"
    r"ceramic|steel|silicone|leather|"
    r"smooth|rough|textured|glossy|matte|shiny|wet|damp|dry|absorbent|porous|"
    r"soft|rigid|fluffy|thick|thin|full|empty|heavy|light)\b", re.IGNORECASE)

_print_lock = threading.Lock()


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", (text or "").lower())}


def removed_cues(observable: str, rewritten: str) -> list[str]:
    """Cue words present in the observable scene but gone from the rewrite."""
    before = {m.group(0).lower() for m in _CUE_WORDS.finditer(observable)}
    after = {m.group(0).lower() for m in _CUE_WORDS.finditer(rewritten)}
    return sorted(before - after)


_NON_DIVERGENT = re.compile(
    r"\b(easily|easy|readily|freely|effortless(?:ly)?|smoothly)\b", re.IGNORECASE)


def scene_similarity(observable: str, rewritten: str) -> float:
    """How much of the observable wording survived. autojunk=False matters: the
    default heuristic treats common characters in long strings as junk and
    reports ~0.03 for what is actually a one-clause edit."""
    import difflib
    return difflib.SequenceMatcher(None, observable, rewritten, autojunk=False).ratio()


def review_flags(rec: dict[str, Any], observable_scene: str) -> list[str]:
    """Everything worth a human glance before this row is used."""
    flags: list[str] = []
    scene = rec["scene"]
    prop = rec["hidden_property"]

    if prop not in PROPERTY_CATEGORIES:
        flags.append(f"hidden_property {prop!r} not in the allowed list")

    if scene.strip() == observable_scene.strip():
        flags.append("scene identical to observable -- no cue removed")
    elif not removed_cues(observable_scene, scene) and not rec.get("scene_overridden"):
        flags.append("no visual cue word disappeared from the scene")

    # The property has to be stated in the scene, not merely recorded in a field.
    val_words = _words(rec["hidden_property_value"]) - {"like", "very", "with", "that", "this"}
    if val_words and not (val_words & _words(scene)):
        flags.append("hidden_property_value does not appear to be stated in the scene")

    value_lower = rec["hidden_property_value"].lower()
    for pat in _CONTRADICTIONS.get(prop, []):
        m = re.search(pat, scene, re.IGNORECASE)
        if m and m.group(0).lower() not in value_lower:
            flags.append(f"scene still says {m.group(0)!r}, which contradicts {prop}")

    action_words = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", rec.get("_action", ""))}
    clash = sorted(set(removed_cues(observable_scene, scene)) & action_words)
    if clash:
        flags.append(f"scene dropped {clash}, but the action still refers to it")

    if rec["expected_outcome"].strip().lower() == rec["failure_signature"].strip().lower():
        flags.append("expected_outcome and failure_signature are identical")

    # A property that makes the action EASIER usually matches what the object
    # would do anyway, so the pair cannot be told apart -- see id 29's first pass
    # ("underfilled with air, very easily compressed") on a plastic bag.
    if _NON_DIVERGENT.search(rec["hidden_property_value"]):
        flags.append("hidden_property_value may not diverge from default behaviour")

    sim = scene_similarity(observable_scene, scene)
    if sim < 0.55:
        flags.append(f"scene rewritten too heavily (similarity {sim:.2f})")

    return flags


def call_model(api_key: str, base_url: str, provider: str, model: str,
               frame_path: str, group: str, task: str, observable_scene: str,
               action: str, temperature: Optional[float],
               forced_category: Optional[str] = None) -> dict[str, Any]:
    import requests

    g = GROUP_GUIDE[group]
    base_query = QUERY_TEMPLATE.format(
        observable_scene=observable_scene, action=action, group=group, task=task,
        categories=", ".join(PROPERTY_CATEGORIES),
        category=forced_category or g["category"],
        alternatives=g["alternatives"], value=g["value"], scene_says=g["scene_says"])
    query = base_query
    image = encode_image(frame_path)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{base_url.rstrip('/')}/query"
    model_params: dict[str, Any] = {"system_prompt": SYSTEM_PROMPT}
    if temperature is not None:
        model_params["temperature"] = temperature

    keys = ["scene", "hidden_property", "hidden_property_value",
            "expected_outcome", "failure_signature"]
    best: Optional[dict[str, Any]] = None
    last_err: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        payload = {
            "endpoint": "vision", "request_source": "override_params",
            "query": query, "image_file": image,
            "model_provider": provider, "model_name": model,
            "model_params": model_params,
            "response_format": {"type": "json"}, "enable_search": False,
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_S)
            if not resp.ok:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            raw = resp.json().get("response")
            if raw is None:
                raise RuntimeError("no 'response' field in reply")
            data = parse_model_json(raw if isinstance(raw, str) else json.dumps(raw), keys)
        except Exception as err:  # noqa: BLE001
            last_err = err
            if attempt == MAX_RETRIES:
                break
            time.sleep(RETRY_BACKOFF_S * attempt)
            continue

        data = {k: str(v).strip() for k, v in data.items()}
        data["hidden_property"] = data["hidden_property"].lower().replace(" ", "_")
        data["_action"] = action
        best = data

        problems = review_flags(data, observable_scene)
        # Only re-ask for faults the model can actually fix by rewriting.
        fixable = [f for f in problems if "not in the allowed list" in f
                   or "no cue removed" in f or "contradicts" in f
                   or "does not appear to be stated" in f
                   or "may not diverge" in f or "rewritten too heavily" in f
                   or "still refers to it" in f]
        if not fixable or attempt == MAX_RETRIES:
            return data
        with _print_lock:
            print(f"      {fixable}; re-asking", file=sys.stderr)
        query = (f"{base_query}\n\nYour previous answer had these problems: "
                 f"{'; '.join(fixable)}. Fix them and return the JSON again. "
                 f"hidden_property must be exactly one of: "
                 f"{', '.join(PROPERTY_CATEGORIES)}.")

    if best is not None:
        return best
    raise RuntimeError(f"failed after {MAX_RETRIES} attempts: {last_err}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", default=DEFAULT_SPEC)
    p.add_argument("--observable", default=DEFAULT_OBSERVABLE)
    p.add_argument("--frames-dir", default=DEFAULT_FRAMES)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--provider", default=DEFAULT_PROVIDER)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default=None)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only-id", default=None, help="comma-separated ids")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    spec = json.load(open(args.spec, encoding="utf-8"))
    groups: dict[str, list[int]] = spec["groups"]
    overrides: dict[str, dict] = spec.get("overrides", {})
    group_of = {i: g for g, ids in groups.items() for i in ids}

    obs = {int(r["id"]): r for r in csv.DictReader(open(args.observable, encoding="utf-8"))
           if r.get("prompt")}

    only = {int(x) for x in args.only_id.split(",")} if args.only_id else None
    wanted = sorted(i for i in group_of if (only is None or i in only))

    absent = [i for i in wanted if i not in obs]
    if absent:
        print(f"no observable prompt for id(s) {absent} -- run generate_testset_csv.py first",
              file=sys.stderr)
        wanted = [i for i in wanted if i in obs]

    os.makedirs(args.cache_dir, exist_ok=True)
    temperature = None if args.temperature < 0 else args.temperature
    print(f"{len(wanted)} unobservable variant(s) to build")

    api_key: Optional[str] = None
    done = 0

    def work(rid: int):
        nonlocal api_key, done
        cache = os.path.join(args.cache_dir, f"{rid:03d}.json")
        if os.path.exists(cache) and not args.overwrite:
            with open(cache, encoding="utf-8") as fh:
                return rid, json.load(fh)
        row = obs[rid]
        ov = overrides.get(str(rid), {})
        if api_key is None:
            api_key = resolve_api_key(args.api_key)
        try:
            data = call_model(api_key, args.base_url, args.provider, args.model,
                              os.path.join(args.frames_dir, row["frame"]),
                              group_of[rid], row["task"], row["scene"], row["action"],
                              temperature, ov.get("hidden_property"))
        except Exception as err:  # noqa: BLE001
            with _print_lock:
                print(f"  [{rid}] FAILED: {err}", file=sys.stderr)
            return rid, None
        data.pop("_action", None)
        if "scene" in ov:
            data["scene_overridden"] = True
        data.update({k: v for k, v in ov.items() if k in data or k in
                     ("scene", "expected_outcome", "failure_signature")})
        rec = {
            "id": rid,
            "task_group": group_of[rid],
            "frame": row["frame"],
            # Copied verbatim -- never generated, so the pair differs only in scene.
            "action": row["action"],
            "observable_scene": row["scene"],
            **data,
            "removed_cues": removed_cues(row["scene"], data["scene"]),
            "scene_similarity": round(scene_similarity(row["scene"], data["scene"]), 3),
            "model": f"{args.provider}/{args.model}",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        rec["review"] = review_flags(data, row["scene"])
        with open(cache, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2, ensure_ascii=False)
        with _print_lock:
            done += 1
            print(f"  [{done}/{len(wanted)}] id {rid} ({rec['hidden_property']}): "
                  f"{rec['hidden_property_value'][:52]}")
        return rid, rec

    results: dict[int, dict] = {}
    if wanted:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for rid, rec in pool.map(work, wanted):
                if rec:
                    results[rid] = rec

    # Fold in every other spec id that already has a cached record, so a
    # targeted --only-id run refreshes those rows instead of truncating the CSV
    # down to just them.
    for rid in sorted(group_of):
        if rid in results or rid not in obs:
            continue
        cache = os.path.join(args.cache_dir, f"{rid:03d}.json")
        if os.path.exists(cache):
            with open(cache, encoding="utf-8") as fh:
                results[rid] = json.load(fh)

    cols = ["id", "prompt_id", "video_file", "task", "task_group", "frame",
            "observability", "hidden_property", "hidden_property_value",
            "scene", "action", "prompt", "expected_outcome", "failure_signature",
            "observable_scene", "removed_cues", "scene_similarity", "review"]
    n_flag = 0
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for rid in sorted(results):
            r = results[rid]
            # Recomputed here, not read from the record: cached rows were written
            # by whatever checks existed at generation time, so trusting their
            # stored verdict would silently exempt them from newer rules.
            r["removed_cues"] = removed_cues(r["observable_scene"], r["scene"])
            r["scene_similarity"] = round(
                scene_similarity(r["observable_scene"], r["scene"]), 3)
            r["review"] = review_flags({**r, "_action": r["action"]}, r["observable_scene"])
            if r["review"] and r.get("scene_overridden"):
                r["review"] = [f for f in r["review"] if "no visual cue" not in f]
            n_flag += bool(r["review"])
            w.writerow({
                "id": rid,
                "prompt_id": f"{rid:03d}_unobs",
                "video_file": obs[rid]["video_file"],
                "task": obs[rid]["task"],
                "task_group": r["task_group"],
                "frame": r["frame"],
                "observability": "unobservable",
                "hidden_property": r["hidden_property"],
                "hidden_property_value": r["hidden_property_value"],
                "scene": r["scene"],
                "action": r["action"],
                "prompt": build_prompt(r["scene"], r["action"]),
                "expected_outcome": r["expected_outcome"],
                "failure_signature": r["failure_signature"],
                "observable_scene": r["observable_scene"],
                "removed_cues": ", ".join(r["removed_cues"]),
                "scene_similarity": r.get("scene_similarity", ""),
                "review": "; ".join(r["review"]),
            })

    # Invariant: the action must be byte-identical to the observable one.
    mismatched = [rid for rid, r in results.items() if r["action"] != obs[rid]["action"]]
    print(f"\nwrote {len(results)} variant(s) -> {args.out}")
    print(f"action identical to observable: {len(results) - len(mismatched)}/{len(results)}")
    if mismatched:
        print(f"ACTION MISMATCH on {mismatched}", file=sys.stderr)
    if n_flag:
        print(f"{n_flag} row(s) carry review flags (see the 'review' column)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

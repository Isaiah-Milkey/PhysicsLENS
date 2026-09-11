#!/usr/bin/env python3
"""Generate video-generation prompts for Xin's testset frames, keyed by index.csv id.

One row per index.csv video id. The prompt is scene + action ONLY -- no expected
outcome, no observability, no taxonomy domain.

The hard constraint: the ACTION must describe the physical motion and must not
allude to its result. "The robot wipes the cloth across the table surface" is
correct; "The robot cleans the table" is not, because the benchmark is testing
whether the video model can infer that wiping a wet surface with a towel makes
it clean.

That constraint fights the source data: Xin's `task` column is goal-phrased
("Wipe the water off the table with a rag"). The task is still passed to the
model, because most frames are frame 0 where the motion has not begun and the
task is the only cue for WHICH motion is coming -- but it is passed with an
explicit instruction and worked example for converting goal -> bare motion, and
every generated action is scanned afterwards for outcome language.

Usage
-----
    python generate_testset_csv.py --limit 3      # try a few first
    python generate_testset_csv.py                # all frames -> testset_prompts.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
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

DEFAULT_INDEX = "physicslens_robot_data/meta/index.csv"
DEFAULT_FRAMES = "testset_frames"
DEFAULT_OUT = "testset_prompts.csv"
DEFAULT_CACHE = "testset_cache"

SYSTEM_PROMPT = """\
You are annotating the FIRST FRAME of a short robot-manipulation video, to build \
a prompt for a video-generation model.

Write plain, neutral, third-person English. Be concrete about objects, materials, \
spatial relations and physically relevant surface properties. Never mention the \
camera, the image, the photo, the frame, or the annotation task: describe the \
scene as a physical situation, not as a picture.

Return only the requested JSON object -- no markdown fences, no commentary."""

QUERY_TEMPLATE = """\
The robot in this frame is about to perform this logged task: "{task}"

Use that ONLY to identify which physical motion is beginning. Do not repeat its \
wording and do not restate its goal.

Produce a JSON object with exactly these keys: "scene", "action".

- scene: 3-5 sentences describing the static setup at this instant. Name the \
objects, their colours, materials and rough sizes, how they are arranged, and \
where each robot arm is relative to them. Then finish with one sentence stating \
the physical properties that matter for what is about to happen -- whether a \
surface is smooth, porous, wet or dry, whether a cloth looks textured and \
absorbent, how full a container is. State only what is visible.

- action: ONE sentence naming the COMPLETE physical manipulation the robot is \
about to carry out -- not merely the first reach towards an object. Most frames \
are captured before the motion starts, so "the robot moves its arm toward the \
bottle" is NOT enough: name the whole movement, including the manipulation the \
task implies. Keep the natural motion verbs -- wipe, pour, push, fold, stack, \
squeeze, tap are all fine, because they name movements. What must be left out \
is the RESULT: the state anything ends up in.

  Task "Grasp the thermos with one hand and pour water into the cup lid."
    CORRECT: "The robot grasps the thermos, lifts it, and tips it over the cup lid."
    WRONG:   "The robot moves its hand toward the thermos."  (only the reach)
    WRONG:   "The robot fills the cup lid with water."        (states the result)

CRITICAL -- the action must NOT state, name or imply the result of the motion. \
The whole point of this benchmark is to test whether a video model can infer the \
consequence on its own, so revealing it destroys the test.

  Task "Wipe the water off the table with a rag."
    CORRECT: "The robot wipes the cloth across the table surface."
    WRONG:   "The robot cleans the table."           (states the result)
    WRONG:   "The robot wipes the water off the table."  (states the result)

  Task "Pour the medicine from the bottle into the box."
    CORRECT: "The robot tips the bottle over the open box."
    WRONG:   "The robot fills the box with medicine."

Note that "wipes" itself is fine -- it is the motion. It is "wipes THE WATER \
OFF" that is forbidden, because the water leaving is the result.

Never use result words such as clean, dry, empty, full, remove, spotless, or \
constructions like "until ...", "so that ...", "resulting in ...", "leaving the \
... clean". Stop the sentence at the motion."""

# Outcome language that would give the answer away. Motion verbs that merely
# share a name with their result (fold, stack, push, pour) are deliberately NOT
# listed -- those describe the movement itself and are fine.
_OUTCOME_PATTERNS = [
    r"\b(clean|cleans|cleaning|cleaned|spotless|dry|dries|drying|dried)\b",
    r"\b(remove[sd]?|removing|empt(?:y|ies|ied)|fills?|filled|filling)\b",
    r"\bwip(?:e|es|ing)\s+(?:away|off|up)\b",
    r"\b(?:until|so that|resulting in|leaving|so as to|in order to)\b",
    r"\b(?:successfully|neatly|thoroughly)\b",
]
_OUTCOME_RE = re.compile("|".join(_OUTCOME_PATTERNS), re.IGNORECASE)


def find_outcome_leaks(action: str) -> list[str]:
    """Return offending snippets if the action gives away its own result."""
    return sorted({m.group(0).strip().lower() for m in _OUTCOME_RE.finditer(action or "")})


def build_prompt(scene: str, action: str) -> str:
    return f"Scene: {scene.strip()}\nAction: {action.strip()}"


_print_lock = threading.Lock()


def call_model(api_key: str, base_url: str, provider: str, model: str,
               frame_path: str, task: str, temperature: Optional[float]) -> dict[str, Any]:
    """One frame -> {"scene", "action"}, retrying on transport and contract errors.

    An action that leaks its outcome is also retried, with the offending phrase
    quoted back -- the same escalation that worked for the tape-marker leak.
    """
    import requests
    import time

    base_query = QUERY_TEMPLATE.format(task=task or "unknown")
    query = base_query
    image = encode_image(frame_path)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{base_url.rstrip('/')}/query"
    model_params: dict[str, Any] = {"system_prompt": SYSTEM_PROMPT}
    if temperature is not None:
        model_params["temperature"] = temperature

    best: Optional[dict[str, Any]] = None
    last_err: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        payload = {
            "endpoint": "vision",
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
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            raw = resp.json().get("response")
            if raw is None:
                raise RuntimeError("no 'response' field in reply")
            data = parse_model_json(raw if isinstance(raw, str) else json.dumps(raw),
                                    ["scene", "action"])
        except Exception as err:  # noqa: BLE001
            last_err = err
            if attempt == MAX_RETRIES:
                break
            time.sleep(RETRY_BACKOFF_S * attempt)
            continue

        best = data
        leaks = find_outcome_leaks(data["action"])
        if not leaks:
            return data
        if attempt == MAX_RETRIES:
            break
        with _print_lock:
            print(f"      outcome leak {leaks}; re-asking", file=sys.stderr)
        query = (
            f"{base_query}\n\nYour previous action sentence was "
            f"{data['action']!r}, which reveals the result via {leaks}. "
            "Rewrite it as the bare physical motion only, stopping before any "
            "consequence or change of state."
        )

    if best is not None:
        return best
    raise RuntimeError(f"failed after {MAX_RETRIES} attempts: {last_err}")


def frame_for(row: dict[str, str], frames_by_id: dict[int, str]) -> Optional[str]:
    return frames_by_id.get(int(row["id"]))


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("--frames-dir", default=DEFAULT_FRAMES)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--provider", default=DEFAULT_PROVIDER)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default=None)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="only process the first N frames")
    p.add_argument("--only-id", default=None, help="comma-separated index ids")
    p.add_argument("--overwrite", action="store_true", help="ignore cached results")
    return run(p.parse_args(argv))


def run(args: argparse.Namespace) -> int:
    rows = list(csv.DictReader(open(args.index, encoding="utf-8")))
    frames = sorted(f for f in os.listdir(args.frames_dir) if not f.startswith("."))

    frames_by_id: dict[int, str] = {}
    for f in frames:
        m = re.match(r"^(\d+)_", f)
        if m:
            frames_by_id[int(m.group(1))] = f

    only = {int(x) for x in args.only_id.split(",")} if args.only_id else None
    os.makedirs(args.cache_dir, exist_ok=True)

    todo = []
    for r in rows:
        rid = int(r["id"])
        if only and rid not in only:
            continue
        fname = frames_by_id.get(rid)
        if fname:
            todo.append((r, fname))
    if args.limit is not None:
        todo = todo[: args.limit]

    missing = [int(r["id"]) for r in rows if frames_by_id.get(int(r["id"])) is None]
    print(f"{len(rows)} index rows, {len(frames)} frames, {len(todo)} to process")
    if missing:
        print(f"no frame for id(s) {missing} -- row(s) kept with an empty prompt")

    temperature = None if args.temperature < 0 else args.temperature
    api_key: Optional[str] = None
    done = 0

    def work(item):
        nonlocal api_key, done
        row, fname = item
        rid = int(row["id"])
        cache = os.path.join(args.cache_dir, f"{rid:03d}.json")
        if os.path.exists(cache) and not args.overwrite:
            with open(cache, encoding="utf-8") as fh:
                return rid, json.load(fh)
        if api_key is None:
            api_key = resolve_api_key(args.api_key)
        try:
            data = call_model(api_key, args.base_url, args.provider, args.model,
                              os.path.join(args.frames_dir, fname), row["task"], temperature)
        except Exception as err:  # noqa: BLE001
            with _print_lock:
                print(f"  [{rid}] FAILED: {err}", file=sys.stderr)
            return rid, None
        rec = {
            "id": rid,
            "frame": fname,
            "scene": str(data["scene"]).strip(),
            "action": str(data["action"]).strip(),
            "model": f"{args.provider}/{args.model}",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        rec["outcome_leaks"] = find_outcome_leaks(rec["action"])
        with open(cache, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2, ensure_ascii=False)
        with _print_lock:
            done += 1
            print(f"  [{done}/{len(todo)}] id {rid}: {rec['action'][:64]}")
        return rid, rec

    results: dict[int, dict] = {}
    if todo:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for rid, rec in pool.map(work, todo):
                if rec:
                    results[rid] = rec

    cols = ["id", "video_file", "task", "frame", "scene", "action", "prompt",
            "action_review", "note"]
    n_written = n_leak = 0
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            rid = int(r["id"])
            # Not skipped for --only-id: fall back to the cached record so a
            # targeted re-run refreshes one row rather than truncating the CSV.
            rec = results.get(rid)
            if rec is None and frames_by_id.get(rid):
                cache = os.path.join(args.cache_dir, f"{rid:03d}.json")
                if os.path.exists(cache):
                    with open(cache, encoding="utf-8") as fh:
                        rec = json.load(fh)
            if rec is None:
                fname = frames_by_id.get(rid)
                w.writerow({"id": rid, "video_file": r["file"], "task": r["task"],
                            "frame": fname or "", "scene": "", "action": "", "prompt": "",
                            "action_review": "",
                            "note": "no frame provided" if not fname else "generation failed"})
                continue
            leaks = rec.get("outcome_leaks") or []
            n_leak += bool(leaks)
            n_written += 1
            w.writerow({
                "id": rid, "video_file": r["file"], "task": r["task"], "frame": rec["frame"],
                "scene": rec["scene"], "action": rec["action"],
                "prompt": build_prompt(rec["scene"], rec["action"]),
                "action_review": "; ".join(leaks),
                "note": "",
            })

    print(f"\nwrote {n_written} prompt(s) -> {args.out}")
    if n_leak:
        print(f"WARNING: {n_leak} action(s) still contain outcome language "
              f"(see action_review column)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

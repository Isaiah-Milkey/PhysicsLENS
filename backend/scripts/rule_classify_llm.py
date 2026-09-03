"""
Re-label the human violated-rules with an LLM instead of keyword matching.

WHY. The keyword taxonomy (rule_taxonomy.py) assigns a category by surface
words, and that demonstrably mislabels: "The bowling pins cannot fall over
without an impacting object" contains "fall" so it lands in GRAVITY, but it is a
statement about CONTACT causing motion — a collision rule. Gravity's ceiling
measured at chance (0.516 with all 55 signals), and taxonomy noise of this kind
is one of the two candidate explanations; the other is genuine co-occurrence
(49% of gravity clips carry a second category).

Those have different fixes, so they need separating. This re-labels each rule
with a text LLM that sees the whole sentence and the category definitions. If
gravity's ceiling rises with cleaner labels, the problem was the instrument. If
it does not, the categories genuinely overlap in this data and no amount of
detector work will separate them.

Note the deliberate reversal: rule_taxonomy.py argues for keyword matching
because the mapping is the measurement instrument and should be deterministic.
That still holds for the reported taxonomy — this script exists to TEST that
instrument against an independent one, not to replace it silently. Both label
sets are kept and compared.

Usage:
  python backend/scripts/rule_classify_llm.py --data data/videophy1200
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from videophy_eval import _retry, client            # noqa: E402
from rule_taxonomy import parse_rules               # noqa: E402

MODEL = "qwen3-235b-a22b-instruct-2507"

CATS = ["collision", "gravity", "deformation", "momentum",
        "friction", "fluid", "permanence", "other"]

PROMPT = """Classify a physics rule that a human annotator said an AI-generated video BROKE.

Categories — pick every one that genuinely applies:
- collision: an effect requires physical contact; contact must occur for the effect
- gravity: unsupported things must fall; nothing floats or hovers without support
- deformation: solid objects keep their shape and structural integrity
- momentum: speed/direction changes only from applied forces; energy is conserved
- friction: surfaces resist sliding; contact dissipates motion
- fluid: liquids, smoke, spray behave as fluids
- permanence: objects do not appear, vanish, duplicate, or turn into other objects
- other: none of the above

Be strict about the DISTINCTION between collision and gravity. A rule about
something falling BECAUSE it was hit, or not falling because nothing hit it, is
COLLISION (it is about contact causing motion), not gravity. Gravity is only for
rules about unsupported objects failing to fall, floating, or hovering.

Rule: "{rule}"

Reply with ONLY a JSON array of category names, e.g. ["collision"] or ["gravity","deformation"]."""


def classify(c, rule):
    r = _retry(lambda: c.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=60,
        messages=[{"role": "user", "content": PROMPT.format(rule=rule[:400])}]))
    txt = (r.choices[0].message.content or "").strip()
    for cand in (txt, txt[txt.find("["):txt.rfind("]") + 1] if "[" in txt else ""):
        try:
            v = json.loads(cand)
            if isinstance(v, list):
                return sorted({str(x).lower() for x in v
                               if str(x).lower() in CATS} - {"other"})
        except Exception:  # noqa: BLE001
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/videophy1200")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    data = ROOT / a.data
    clips = json.loads((data / "manifest.json").read_text())["clips"]

    todo = [(c["clip_id"], " ; ".join(parse_rules(c.get("violated_rules", ""))))
            for c in clips if len(c.get("violated_rules") or "") > 4]
    print(f"classifying {len(todo)} rule annotations with {MODEL}", flush=True)

    c = client()
    out = {}
    done = [0]

    def work(t):
        cid, rule = t
        try:
            out[cid] = classify(c, rule)
        except Exception as e:  # noqa: BLE001
            print(f"  fail {cid[:30]}: {str(e)[:60]}", file=sys.stderr)
            out[cid] = None
        done[0] += 1
        if done[0] % 100 == 0:
            print(f"    {done[0]}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, todo))

    per_clip = {c["clip_id"]: (out.get(c["clip_id"]) or []) for c in clips}
    from collections import Counter
    cnt = Counter(x for v in per_clip.values() for x in v)
    ok = sum(1 for v in out.values() if v is not None)
    print(f"\n  classified {ok}/{len(todo)}")
    print("  counts:", dict(cnt.most_common()))

    # agreement with the keyword taxonomy
    kw = json.loads((data / "rule_categories.json").read_text())["per_clip"]
    both = [(set(kw.get(k, [])), set(v)) for k, v in per_clip.items() if v]
    exact = sum(1 for x, y in both if x == y)
    olap = sum(1 for x, y in both if x & y)
    print(f"  agreement with keyword taxonomy: exact {exact}/{len(both)}, "
          f"any-overlap {olap}/{len(both)}")

    outp = data / "rule_categories_llm.json"
    outp.write_text(json.dumps({"model": MODEL, "counts": dict(cnt),
                                "per_clip": per_clip}, indent=1))
    print(f"  -> {outp}")


if __name__ == "__main__":
    main()

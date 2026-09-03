"""
Domain-knowledge probe batteries — what a physicist would actually check.

WHY THIS IS NOT A REPEAT. The earlier probes asked ONE abstract question per
category ("is any object unsupported yet not falling"). Five attempts to lift
the weak specialists failed, and a ceiling test over 55 engineered signals plus
768 embedding dims put gravity at 0.516. But that ceiling only covers signals
already measured — a battery of NEW, specific questions is genuinely new
measurement, not a recombination of the old ones.

The change is decomposition plus concreteness. Instead of "does gravity look
wrong", ask the individual things that make it wrong:

  - free fall must ACCELERATE (later gaps bigger than earlier gaps)
  - an unsupported object must not hold its height
  - nothing rises without a visible driver
  - landings dissipate: bounce, squash, or settle — never a dead stop
  - support must be plausible: not air, not something far too weak

Each is a separate yes/no with its own token-probability score, so a category
becomes a 5-dimensional measurement instead of a scalar. Weak-but-real
sub-signals can then combine even when the holistic question is uninformative.

Run over the 704 rule-annotated clips only — that set IS the discriminative
comparison (positives plus other-category violators), so scoring the other 496
clean clips would spend budget without sharpening any AUC.

Usage:
  python backend/scripts/domain_probes.py --cats gravity,momentum
  python backend/scripts/domain_probes.py --cats gravity,momentum,deformation,permanence
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from videophy_eval import _token_probs, client        # noqa: E402
from physics_probes import frames                     # noqa: E402
import base64 as _b64
import numpy as _np


def frames_ordered(data, clip, k=8, order="temporal"):
    """k frames in the frozen temporal OR shuffled ordering.

    The shuffled ordering is the one fixed at staging time, so the control is
    reproducible and identical to the ablation used on the holistic score.
    """
    if order == "temporal":
        return frames(data, clip, k)
    idx = clip["order_shuffled"]
    if k < len(idx):
        idx = [idx[i] for i in _np.linspace(0, len(idx) - 1, k).astype(int)]
    d = data / "frames" / clip["clip_id"]
    out = []
    for i in idx:
        f = d / f"{i:03d}.jpg"
        if f.exists():
            out.append("data:image/jpeg;base64," + _b64.b64encode(f.read_bytes()).decode())
    return out
from gepa_optimize import load                        # noqa: E402

MODEL = "gemma4-31b-it"
_P = ("Look at these {n} frames, sampled in order from a video of: \"{caption}\".\n")

# DEGREE, not yes/no. These questions are specific enough that the model answers
# "No" confidently and "Yes" falls outside the top-20 logprobs entirely — so
# p.get("yes", 0.0) returns the default and EVERY clip ties at exactly 0.0, with
# no ranking information whatever. Verified on a 3-clip smoke run: 13 of 15
# sub-scores were exactly 0.0. A 1-5 degree scale spreads the mass over five
# tokens that are all in-distribution, which is the same fix that gave the
# likert prompt 300/300 distinct values.
_A = ("\nHow clearly does this happen in these frames?\n"
      "1 = not at all, 2 = barely, 3 = somewhat, 4 = clearly, 5 = blatantly.\n"
      "Answer with exactly one digit, 1 to 5.")

# "Yes" always means THE DEFECT IS PRESENT, so sub-scores combine without sign
# bookkeeping.
BATTERIES = {
    "gravity": [
        ("g_no_accel",
         "Find anything falling. Real falling speeds UP: the gap it moves "
         "between later frames must be LARGER than between earlier frames.\n"
         "Does something fall at a constant speed, or drift down too slowly, "
         "instead of accelerating?"),
        ("g_holds_height",
         "Is there an object in the air that keeps the SAME height across "
         "several frames, when nothing is holding it up?"),
        ("g_rises",
         "Does anything move upward without a visible cause — no throw, no jet, "
         "no lift, no one pushing it?"),
        ("g_dead_landing",
         "When something lands or comes to rest, does it stop dead — with no "
         "bounce, no squash, no wobble, no settling of any kind?"),
        ("g_bad_support",
         "Is anything resting on empty space, or held up by something far too "
         "thin or weak to support its weight?"),
    ],
    "momentum": [
        ("m_gains_energy",
         "After a hit, bounce or collision, does anything end up moving FASTER "
         "than it was before? A bounce must always return less speed."),
        ("m_stops_dead",
         "Does a moving object stop suddenly without hitting anything and "
         "without slowing down first?"),
        ("m_no_recoil",
         "When one object strikes another, does the STRIKING object continue "
         "as if nothing happened — no slowing, no bounce back, no recoil?"),
        ("m_midair_turn",
         "Does anything change direction while in mid-air, with nothing "
         "touching it?"),
        ("m_mass_mismatch",
         "Does a heavy object get thrown or deflected by something much "
         "lighter, or a light object shrug off a heavy impact?"),
    ],
    "deformation": [
        ("d_shape_drift",
         "Does a solid, rigid object slowly change its shape, length or "
         "proportions across the frames while nothing is squeezing it?"),
        ("d_bend_no_force",
         "Does something bend, twist or fold where no force is being applied "
         "to bend it?"),
        ("d_no_deform_on_impact",
         "At a hard impact, do the objects stay perfectly rigid — no squash, "
         "dent, compression or give at all, where real materials would show it?"),
        ("d_merge",
         "Do two separate objects blend, fuse or pass into one another as if "
         "they were made of liquid?"),
        ("d_texture_swim",
         "Does the surface pattern, print or texture on an object slide around "
         "or crawl across it instead of staying fixed to the surface?"),
    ],
    "permanence": [
        ("p_vanish",
         "Does any object, limb or body part disappear between frames?"),
        ("p_appear",
         "Does any object, limb or body part appear out of nowhere?"),
        ("p_count_change",
         "Does the NUMBER of a repeated thing change — fingers, legs, wheels, "
         "people, objects in a set?"),
        ("p_identity_swap",
         "Does something turn into a different object, or swap its identity, "
         "between frames?"),
        ("p_occlusion_fail",
         "When something passes behind another object, does it fail to come "
         "back out correctly — wrong place, wrong shape, or not at all?"),
    ],
    "collision": [
        ("c_gap_at_impact",
         "At the moment of contact, is there still a visible GAP between the "
         "two objects?"),
        ("c_early_reaction",
         "Does the target start reacting BEFORE it is actually touched?"),
        ("c_pass_through",
         "Does anything pass through a solid object instead of being stopped "
         "by it?"),
        ("c_no_contact_effect",
         "Does the intended effect happen even though the two things never "
         "visibly meet?"),
    ],
    "friction": [
        ("f_slides_forever",
         "Does something slide or roll along a surface without slowing down at "
         "all?"),
        ("f_grip_fail",
         "Does something slip on a surface where it should grip, or grip where "
         "it should slip?"),
        ("f_no_rolling_link",
         "Does a wheel or ball move across the ground without its spin matching "
         "how far it travels — sliding rather than rolling?"),
    ],
    "fluid": [
        ("fl_vanish",
         "Does liquid, spray or smoke appear or vanish instantly instead of "
         "flowing, draining or dispersing?"),
        ("fl_shape",
         "Does a fluid hold a shape it could not hold — a rigid blob, a frozen "
         "splash, a wall of water standing up?"),
        ("fl_no_splash",
         "Does something enter or strike liquid without the splash, ripple or "
         "disturbance it should cause?"),
    ],
}

# ── v2 batteries ──────────────────────────────────────────────────────────────
# Designed from what actually separated in v1. Across all 30 sub-checks, the
# ones that worked asked the model to COMPARE two things it can see, or COUNT
# something discrete:
#     g_no_accel     0.661  compare gap sizes between frames
#     p_count_change 0.665  count a repeated thing
# The ones at chance asked it to judge an ABSENCE or ATTRIBUTE A CAUSE:
#     g_rises        0.498  "without a visible cause"
#     d_no_deform    0.497  "no squash where real materials would show it"
#     m_no_recoil    0.458  "as if nothing happened"
#     c_gap_at_impact 0.491 requires precise timing perception
# momentum, deformation and collision were written entirely in the failing
# style. These rewrite them as comparisons and counts.
BATTERIES_V2 = {
    "momentum_v2": [
        ("m2_faster_after",
         "Compare how far the moving object travels between frames BEFORE the "
         "hit versus AFTER it. Does it cover MORE distance per frame after the "
         "hit than before?"),
        ("m2_speed_jump",
         "Compare the spacing of the object between each pair of frames. Is "
         "there one place where the spacing suddenly jumps much larger, with no "
         "other object touching it at that point?"),
        ("m2_stops_early",
         "Compare the object's spacing across the last few frames. Does the "
         "spacing go from large straight to zero in a single step, rather than "
         "shrinking gradually?"),
        ("m2_both_move",
         "After two things meet, compare how much EACH of them moves. Does only "
         "one of them change its motion while the other keeps going exactly as "
         "before?"),
    ],
    "deformation_v2": [
        ("d2_length_change",
         "Pick one solid object. Compare its length end-to-end in the first "
         "frame and in the last frame. Has that length visibly changed?"),
        ("d2_straight_edges",
         "Count the straight edges or sharp corners on the main solid object in "
         "the first frame, then count them again in the last frame. Do the "
         "counts differ?"),
        ("d2_proportion",
         "Compare the width-to-height proportion of the main object between the "
         "first and last frame. Has that proportion changed?"),
        ("d2_impact_squash",
         "Compare the object's shape in the frame just before contact and the "
         "frame just after. For a hard impact, is it EXACTLY the same shape?"),
    ],
    "collision_v2": [
        ("c2_gap_size",
         "At the frame where the two objects are closest, compare the gap "
         "between them to the size of the smaller object. Is the gap a "
         "noticeable fraction of that object rather than zero?"),
        ("c2_order",
         "Compare WHEN the target starts moving to WHEN the two objects touch. "
         "Does the target start moving in an EARLIER frame than the touch?"),
        ("c2_overlap",
         "At the closest frame, do the two objects visibly OVERLAP — occupying "
         "the same space — rather than meeting at their surfaces?"),
        ("c2_count_contacts",
         "Count how many times the two objects actually touch across these "
         "frames. Is that count ZERO even though an effect clearly happens?"),
    ],
}
BATTERIES.update(BATTERIES_V2)

# The single best-discriminating question per category, from the v1/v2 sweep.
# Fluid reached 0.944 by fusing the SAME question across three VLMs, while
# fusing seven models on the vague holistic question gained nothing — a specific
# question makes model noise independent, a vague one leaves shared bias. These
# winners were only ever run on gemma4, so the ensemble trick is untested
# exactly where it is most needed.
BATTERIES["winners"] = [
    ("w_gravity",     dict(BATTERIES["gravity"])["g_no_accel"]),
    ("w_permanence",  dict(BATTERIES["permanence"])["p_count_change"]),
    ("w_collision",   dict(BATTERIES["collision_v2"])["c2_overlap"]),
    ("w_deformation", dict(BATTERIES["deformation_v2"])["d2_length_change"]),
    ("w_friction",    dict(BATTERIES["friction"])["f_no_rolling_link"]),
    ("w_momentum",    dict(BATTERIES["momentum"])["m_gains_energy"]),
]

def score_clip(c, data, clip, probes, model=MODEL, caption=True,
               order="temporal", nframes=8):
    imgs = frames_ordered(data, clip, nframes, order)
    if caption:
        head = _P.format(n=len(imgs), caption=clip.get("caption", "")[:200])
    else:
        # Caption-free control: every domain question embeds the clip caption,
        # so a detector could be scoring the PROMPT TEXT rather than the pixels.
        head = (f"Look at these {len(imgs)} frames, sampled in order from a "
                "video.\n")
    out = {}
    for name, q in probes:
        try:
            p = _token_probs(c, model, imgs, head + q + _A)
            # len(k)==1 guard: "" is a substring of every string, and the gateway
            # does emit empty tokens, so a bare `k in "12345"` lets int("") raise.
            mass = {int(k): v for k, v in p.items() if len(k) == 1 and k in "12345"}
            if sum(mass.values()) > 1e-4:
                # Score = 1 - P("1"), i.e. P(the defect is present to ANY degree).
                # Not the expected value over digits: when the model is confident
                # the answer is "1", digits 2-5 fall outside the top-20 logprobs,
                # the EV comes back exactly 1.0, and the clip pins to 0.0 with no
                # gradation (55% of sub-scores did this on a smoke run). P("1") is
                # a true probability from the raw distribution, so 1 - P("1") stays
                # precise however hard the tail is truncated, and it is monotonic
                # in "how much defect the model sees" — which is all AUC needs.
                out[name] = float(min(max(1.0 - p.get("1", 0.0), 0.0), 1.0))
        except Exception as e:  # noqa: BLE001
            print(f"    {name} fail {clip['clip_id'][:26]}: {str(e)[:45]}",
                  file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/videophy1200")
    ap.add_argument("--cats", default="gravity,momentum,deformation,permanence")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--model", default=MODEL,
                    help="gateway model id; the question set is identical across "
                         "models so the ensemble stays apples-to-apples")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-caption", action="store_true",
                    help="strip the caption from the prompt (ablation)")
    ap.add_argument("--order", default="temporal",
                    choices=["temporal", "shuffled"],
                    help="frame order; shuffled is the motion-blindness control")
    ap.add_argument("--frames", type=int, default=8)
    a = ap.parse_args()

    data, clips = load(a.data)
    # rule-annotated clips only: that is exactly the discriminative set
    clips = [c for c in clips if len(c.get("violated_rules") or "") > 4]
    if a.limit:
        clips = clips[:a.limit]

    cats = [x.strip() for x in a.cats.split(",") if x.strip() in BATTERIES]
    probes = [(nm, q) for cat in cats for nm, q in BATTERIES[cat]]
    print(f"domain probes: {len(cats)} categories, {len(probes)} sub-probes, "
          f"{len(clips)} clips = {len(probes)*len(clips)} calls", flush=True)

    c = client()
    t0 = time.time()
    res = [None] * len(clips)
    done = [0]

    def work(i):
        res[i] = score_clip(c, data, clips[i], probes, a.model,
                            caption=not a.no_caption, order=a.order,
                            nframes=a.frames)
        done[0] += 1
        if done[0] % 50 == 0:
            el = time.time() - t0
            print(f"    {done[0]}/{len(clips)} ({el:.0f}s, eta "
                  f"{el/done[0]*(len(clips)-done[0])/60:.0f}m)", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, range(len(clips))))

    out = {cl["clip_id"]: r for cl, r in zip(clips, res) if r}
    tag = '_'.join(cats) + ('' if a.model == MODEL else f"__{a.model}")
    if a.no_caption:
        tag += "__nocap"
    if a.order != "temporal":
        tag += f"__{a.order}"
    if a.frames != 8:
        tag += f"__f{a.frames}"
    outp = data / f"domain_probes_{tag}.json"
    outp.write_text(json.dumps({"model": MODEL, "cats": cats, "n": len(out),
                                "probe_scores": out}, indent=1))
    print(f"\n  {len(out)}/{len(clips)} clips ({time.time()-t0:.0f}s)")
    print(f"  -> {outp}")


if __name__ == "__main__":
    main()

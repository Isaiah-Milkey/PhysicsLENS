"""
Verify whether OPEN-SOURCE VLMs can characterize video-model performance and
identify failure modes — across three architecturally-distinct model families.

Models (all open-weight, run locally on GPU):
  • Qwen/Qwen2.5-VL-7B-Instruct
  • OpenGVLab/InternVL3-8B-hf
  • HuggingFaceTB/SmolVLM2-2.2B-Instruct

Protocol: multi-frame (8 frames in temporal order, one holistic physics judgement),
which an earlier single-vs-multi study showed is the configuration that works.
Each model scores 5 AI-generated vs 5 real physics videos. We report, per model:
  • AI vs real mean suspicion
  • rank AUC (P[AI scored > real]) — the separation metric
  • the failure-mode labels produced
Results -> scripts/vlm_multimodel_results.json
"""
import gc, json, re, sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from tools.video import load_frames, sample_frames

MODELS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "OpenGVLab/InternVL3-8B-hf",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
]

VIDEOS = {
    "ai": [
        "test_videos/ai/sora/basketball-explosion.mp4",
        "test_videos/ai/sora/petri-dish-pandas.mp4",
        "test_videos/ai/sora/ships-in-coffee.mp4",
        "test_videos/ai/lumiere/beer_pouring.mp4",
        "test_videos/ai/videopoet/tidal_wave.mp4",
    ],
    "real": [
        "test_videos/real/physics_iq/balls-collide.mp4",
        "test_videos/real/physics_iq/ball-ramp.mp4",
        "test_videos/real/physics_iq/ball-and-block-fall.mp4",
        "test_videos/real/physics_iq/block-domino.mp4",
        "test_videos/real/wikimedia/bouncing_ball.webm",
    ],
}

PROMPT = (
    "You are a physics expert. These {n} frames are sampled in temporal order from "
    "one video. Judge whether the MOTION and INTERACTIONS across frames obey "
    "real-world physics (gravity, momentum, object permanence, rigid-body and fluid "
    "behavior). AI-generated videos often show objects that morph, appear/disappear, "
    "move without forces, or deform implausibly.\n"
    "Reply with VALID JSON ONLY - no markdown fences:\n"
    '{{"suspicion_score": <float 0.0-1.0>, "suspected_failure": "<brief label or null>", '
    '"explanation": "<one or two sentences>", "confidence": <float 0.0-1.0>}}'
)


def to_pil(frame_bgr, max_side=512):
    h, w = frame_bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        frame_bgr = cv2.resize(frame_bgr, (int(w * s), int(h * s)))
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def parse_json(raw):
    raw = raw.replace("{{", "{").replace("}}", "}")  # some models echo literal {{ }}
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {"suspicion_score": None, "suspected_failure": None,
                "explanation": f"unparseable: {raw[:100]}", "confidence": 0.0}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        # tolerate trailing junk / minor issues
        try:
            return json.loads(m.group(0)[:m.group(0).rfind("}") + 1])
        except Exception:
            return {"suspicion_score": None, "suspected_failure": None,
                    "explanation": f"badjson: {raw[:100]}", "confidence": 0.0}


def ask(model, processor, images, prompt):
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    raw = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0]
    return parse_json(raw)


def rank_auc(ai, real):
    pairs = [(a, r) for a in ai for r in real if a is not None and r is not None]
    if not pairs:
        return float("nan")
    wins = sum(1.0 if a > r else 0.5 if a == r else 0.0 for a, r in pairs)
    return round(wins / len(pairs), 3)


def eval_model(model_id, clips):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print(f"\n### Loading {model_id}", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    rows = []
    for label, name, imgs in clips:
        r = ask(model, processor, imgs, PROMPT.replace("{n}", str(len(imgs))))
        s = r.get("suspicion_score")
        s = float(s) if isinstance(s, (int, float)) else None
        rows.append({"label": label, "video": name, "score": s,
                     "failure": r.get("suspected_failure"),
                     "explanation": r.get("explanation")})
        print(f"  [{label:4s}] {name:28s} score={s} fail={r.get('suspected_failure')}", flush=True)

    del model, processor
    gc.collect(); torch.cuda.empty_cache()

    ai = [r["score"] for r in rows if r["label"] == "ai"]
    re_ = [r["score"] for r in rows if r["label"] == "real"]
    ai_v = [x for x in ai if x is not None]; re_v = [x for x in re_ if x is not None]
    summary = {
        "ai_mean": round(float(np.mean(ai_v)), 3) if ai_v else None,
        "real_mean": round(float(np.mean(re_v)), 3) if re_v else None,
        "auc": rank_auc(ai, re_),
        "n_parsed": len(ai_v) + len(re_v),
    }
    print(f"  => AI {summary['ai_mean']} vs real {summary['real_mean']} | AUC {summary['auc']}", flush=True)
    return {"rows": rows, "summary": summary}


def main():
    # Pre-load all clips once (8 frames each)
    clips = []
    for label, paths in VIDEOS.items():
        for rel in paths:
            vp = ROOT / rel
            if not vp.exists():
                print(f"SKIP {rel}", flush=True); continue
            frames, _ = load_frames(str(vp))
            if not frames:
                print(f"SKIP unreadable {rel}", flush=True); continue
            imgs = [to_pil(f) for f in sample_frames(frames, 8)]
            clips.append((label, Path(rel).name, imgs))
    print(f"Loaded {len(clips)} clips.", flush=True)

    out = {"protocol": "multi-frame (8 frames, temporal order)", "models": {}}
    for mid in MODELS:
        try:
            out["models"][mid] = eval_model(mid, clips)
        except Exception as exc:
            print(f"  !! {mid} FAILED: {exc}", flush=True)
            out["models"][mid] = {"error": str(exc)}

    Path(__file__).parent.joinpath("vlm_multimodel_results.json").write_text(json.dumps(out, indent=2))
    print("\n==== SEPARATION SUMMARY (AUC = P[AI scored more suspicious than real]) ====", flush=True)
    for mid, res in out["models"].items():
        s = res.get("summary")
        if s:
            print(f"  {mid:42s} AI={s['ai_mean']} real={s['real_mean']} AUC={s['auc']} (n={s['n_parsed']}/10)", flush=True)
        else:
            print(f"  {mid:42s} ERROR: {res.get('error')}", flush=True)


if __name__ == "__main__":
    main()

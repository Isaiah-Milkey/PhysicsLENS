"""
PhysicsLENS Physics Diagnostic — FastAPI Backend
------------------------------------------------
Pipeline event schema yielded by each run() generator:
  {"type": "log",      "level": "info|warn|error|success", "text": "..."}
  {"type": "metric",   "label": "...", "value": "...", "sub": "..."}
  {"type": "severity", "label": "...", "value": 0-100, "color": "#hex"}
  {"type": "image",    "data": "<base64>", "mime": "image/png", "caption": "..."}
  {"type": "signal",   "source": "...", "fps": float, "signals": [{"frame", "signal_type", "score"}]}
  {"type": "marker_video", "data": "<base64>", "mime": "video/mp4", "fps": float,
                       "duration": float, "src_width": int, "src_height": int,
                       "markers": [{"id", "t_start", "t_end", "t_center", "label",
                                    "severity", "color", "region"}], "caption": "..."}
  {"type": "done"}
  {"type": "error",    "text": "..."}
"""
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from tools.costs import instrument

# ── Stage 1 — Screening ───────────────────────────────────────────────────────
from pipelines.stage1.temporal_smoothness         import run as run_s1_temporal
from pipelines.stage1.optical_flow_irregularities import run as run_s1_optical_flow
from pipelines.stage1.embedding_biomarkers         import run as run_s1_embeddings
from pipelines.stage1.vlm_suspicion                import run as run_s1_vlm
from pipelines.stage1.camera_motion                import run as run_s1_camera_motion

# ── Stage 2 — Failure Localisation & Hypothesis Testing ──────────────────────
from pipelines.stage2.object_tracker              import run as run_s2_object_tracker
from pipelines.stage2.trajectory_extractor        import run as run_s2_trajectory_extractor
from pipelines.stage2.event_localizer             import run as run_s2_event_localizer
from pipelines.stage2.physics_hypothesis_generator import run as run_s2_hypothesis_generator

# ── Stage 3 — Specialist Evaluation ──────────────────────────────────────────
from pipelines.stage3.collision_specialist    import run as run_s3_collision
from pipelines.stage3.gravity_specialist      import run as run_s3_gravity
from pipelines.stage3.momentum_specialist     import run as run_s3_momentum
from pipelines.stage3.friction_specialist     import run as run_s3_friction
from pipelines.stage3.deformation_specialist  import run as run_s3_deformation
from pipelines.stage3.fluid_specialist        import run as run_s3_fluid
from pipelines.stage3.causality_specialist    import run as run_s3_causality
from pipelines.stage3.consistency_specialist  import run as run_s3_consistency

# ── Stage 4 — Final Diagnosis & Treatment Plan ────────────────────────────────
from pipelines.stage4.physics_consistency_scorer import run as run_s4_scorer
from pipelines.stage4.severity_assessor          import run as run_s4_severity
from pipelines.stage4.physics_breakdown_timer    import run as run_s4_pbt
from pipelines.stage4.failure_explainer          import run as run_s4_explainer
from pipelines.stage4.diagnostic_report          import run as run_s4_report

from tools.vlm_router import model_options as _vlm_options, DEFAULT_MODEL_KEY as _VLM_DEFAULT

# ── Shared VLM setting builders ──────────────────────────────────────────────
# Every VLM-using pipeline offers ONE provider-tagged model dropdown (the value
# encodes CreateAI vs OpenRouter) and ONE API-key field whose meaning follows
# the selected model's provider (blank → the provider's .env credential).
_LOCAL_VLM_OPTIONS = [
    {"value": "qwen2.5-vl-7b", "label": "Qwen2.5-VL 7B — local, no key, AUC 0.92 (recommended)"},
    {"value": "internvl3-8b",  "label": "InternVL3 8B — local, no key, AUC 0.70"},
    {"value": "smolvlm2-2.2b", "label": "SmolVLM2 2.2B — local, no key, AUC 0.50 (fast)"},
]


def _vlm_key_setting():
    return {"id": "api_key", "type": "password", "default": "",
            "label": "API key — OpenRouter key or CreateAI token matching the "
                     "selected model (blank = use .env)"}


def _vlm_model_setting(label="Vision model", *, include_local=False, default=_VLM_DEFAULT):
    opts = (_LOCAL_VLM_OPTIONS + _vlm_options()) if include_local else list(_vlm_options())
    return {"id": "model", "label": label, "type": "select",
            "default": default, "options": opts}


def _createai_key_setting():
    """CreateAI-only key field (no provider choice — used where only CreateAI
    is wired, e.g. the Diagnostic Report's text-only LLM summary)."""
    return {"id": "api_key", "type": "password", "default": "",
            "label": "CreateAI token (blank = use .env CREATEAI_TOKEN)"}


def _naming_model_setting():
    return {"id": "naming_model", "label": "Object-naming model", "type": "select",
            "default": "local",
            "options": [{"value": "local", "label": "Local Qwen2.5-VL (no key)"}]
                       + list(_vlm_options())}

# ---------------------------------------------------------------------------
# Pipeline registry
# ---------------------------------------------------------------------------
# Required keys: id, name, desc, badge, requires_pair, run
# Optional keys: requires_prompt, settings (list of setting defs)
#
# Setting definition:
#   {"id": str, "label": str, "type": "number|text|select|password",
#    "default": any, "min": num, "max": num,
#    "options": [{"value": any, "label": str}]}   # for select type
# ---------------------------------------------------------------------------

PIPELINES = {
    # ── Stage 1: Screening ───────────────────────────────────────────────────
    "s1_temporal": {
        "id":    "s1_temporal",
        "name":  "Temporal Smoothness Anomalies",
        "desc":  "Track keypoints frame-to-frame; compute velocity & acceleration; flag sudden jumps.",
        "badge": "cheap",
        "dummy": False,
        "requires_pair": False,
        "settings": [
            {"id": "num_keypoints",   "label": "Number of keypoints",           "type": "number",
             "default": 5,   "min": 1,   "max": 50},
            {"id": "accel_threshold", "label": "Acceleration threshold (px/s²)", "type": "number",
             "default": 40000, "min": 0.1, "max": 100000},
        ],
        "run": run_s1_temporal,
    },
    "s1_optical_flow": {
        "id":    "s1_optical_flow",
        "name":  "Optical Flow Irregularities",
        "desc":  "Detect flow magnitude spikes, directional chaos, and impossible motion via sparse LK flow.",
        "badge": "cheap",
        "dummy": False,
        "requires_pair": False,
        "settings": [
            {"id": "num_keypoints",         "label": "Tracked keypoints",               "type": "number",
             "default": 30,   "min": 5,   "max": 200},
            {"id": "spike_threshold",       "label": "Flow spike threshold (px/frame)",  "type": "number",
             "default": 20.0, "min": 1.0, "max": 200.0},
            {"id": "consistency_threshold", "label": "Direction consistency floor (0–1)", "type": "number",
             "default": 0.5,  "min": 0.0, "max": 1.0},
        ],
        "run": run_s1_optical_flow,
    },
    "s1_embeddings": {
        "id":    "s1_embeddings",
        "name":  "Embedding Biomarkers",
        "desc":  "Encode whole frames with a vision model; detect latent velocity/acceleration spikes as physics anomalies.",
        "badge": "cheap",
        "dummy": False,
        "requires_pair": False,
        "settings": [
            {"id": "model", "label": "Vision model", "type": "select",
             "default": "dinov2",
             "options": [
                 {"value": "dinov2",  "label": "DINOv2 (facebook/dinov2-base, ViT-B/14)"},
                 {"value": "clip",    "label": "CLIP (openai/clip-vit-base-patch32)"},
                 {"value": "siglip",  "label": "SigLIP (google/siglip-base)"},
             ]},
            {"id": "sample_every",    "label": "Sample every N frames",        "type": "number",
             "default": 2,   "min": 1,  "max": 60},
            {"id": "accel_threshold", "label": "Latent acceleration threshold", "type": "number",
             "default": 0.5, "min": 0.01, "max": 10.0},
        ],
        "run": run_s1_embeddings,
    },
    "s1_camera_motion": {
        "id":    "s1_camera_motion",
        "name":  "Camera Motion Detector",
        "desc":  "KLT optical flow + camera-path decomposition. Detects excessive or erratic camera movement — smooth pans, orbits and zooms score near zero even when fast.",
        "badge": "cheap",
        "dummy": False,
        "requires_pair": False,
        "settings": [
            {"id": "num_points",       "label": "Feature points to track (total)",                                       "type": "number",
             "default": 300,  "min": 20,   "max": 600},
            {"id": "n_zones",          "label": "Zone grid size N (NxN grid, e.g. 3 = 9 zones)",                        "type": "number",
             "default": 3,    "min": 1,    "max": 6},
            {"id": "smooth_frames",    "label": "Temporal smoothing window (frames)",                                    "type": "number",
             "default": 15,   "min": 3,    "max": 60},
            {"id": "motion_threshold", "label": "Motion threshold (0–1) — score above this = excessive movement",         "type": "number",
             "default": 0.60, "min": 0.05, "max": 0.95},
            {"id": "jitter_px",        "label": "Tremor calibration (px/frame of high-freq jitter = score 1.0) — raise to be less sensitive", "type": "number",
             "default": 2.0,  "min": 0.2,  "max": 10.0},
            {"id": "jitter_weight",    "label": "Jitter signal weight (rest goes to model-residual signal)",            "type": "number",
             "default": 0.80, "min": 0.0,  "max": 1.0},
            {"id": "min_inlier_ratio", "label": "Min RANSAC inlier ratio — frames below this skip the residual signal", "type": "number",
             "default": 0.30, "min": 0.05, "max": 0.95},
            {"id": "max_height",       "label": "Max resolution height (0 = no limit)",                                 "type": "number",
             "default": 720,  "min": 0,    "max": 2160},
        ],
        "run": run_s1_camera_motion,
    },
    "s1_vlm": {
        "id":    "s1_vlm",
        "name":  "VLM Physics Anomaly Detector",
        "desc":  "A vision-language model judges the motion across keyframes and extracts concrete physics violations — which law is broken, what is observed, and why it's impossible. Local open-weight models run on the GPU with no API key; API models need an OpenRouter key.",
        "badge": "medium",
        "dummy": False,
        "requires_pair": False,
        "settings": [
            _vlm_model_setting("Model", include_local=True, default="qwen2.5-vl-7b"),
            {"id": "num_frames", "label": "Keyframes to sample", "type": "number",
             "default": 8, "min": 2, "max": 20},
            _vlm_key_setting(),
        ],
        "run": run_s1_vlm,
    },

    # ── Stage 2: Failure Localisation & Hypothesis Testing ───────────────────
    "s2_object_tracker": {
        "id":    "s2_object_tracker",
        "name":  "Object Tracker",
        "desc":  "Segment & name objects with SAM3 (VLM-named concepts), track them, and measure DINOv2 appearance drift. Outputs a labeled segmented video; publishes masks to the evidence bus for the Stage 3 Consistency & Collision specialists. Falls back to Shi-Tomasi+LK if no GPU.",
        "badge": "medium",
        "dummy": False,
        "requires_pair": False,
        "settings": [
            {"id": "method",        "label": "Tracking method",        "type": "select",
             "default": "sam3",
             "options": [
                 {"value": "sam3", "label": "SAM3 segmentation (named objects, GPU)"},
                 {"value": "lk",   "label": "Shi-Tomasi + LK corners (CPU fallback)"},
             ]},
            {"id": "num_frames",    "label": "Max frames (SAM3)",      "type": "number",
             "default": 48, "min": 8, "max": 120},
            {"id": "use_vlm_naming","label": "Name objects with VLM",  "type": "select",
             "default": "true",
             "options": [
                 {"value": "true",  "label": "Enabled (a VLM names the scene)"},
                 {"value": "false", "label": "Disabled (use concept vocabulary)"},
             ]},
            _naming_model_setting(),
            _vlm_key_setting(),
            {"id": "concepts",      "label": "Concepts (optional, comma-separated)", "type": "text",
             "default": "", "placeholder": "e.g. ball, wooden block, cup"},
            {"id": "use_dinov2",    "label": "DINOv2 appearance drift", "type": "select",
             "default": "true",
             "options": [
                 {"value": "true",  "label": "Enabled (requires torch)"},
                 {"value": "false", "label": "Disabled (faster)"},
             ]},
            {"id": "render_video",  "label": "Labeled segmented video", "type": "select",
             "default": "true",
             "options": [
                 {"value": "true",  "label": "Show annotated video"},
                 {"value": "false", "label": "Skip (faster)"},
             ]},
            {"id": "num_keypoints", "label": "Keypoints (LK method)",   "type": "number",
             "default": 50, "min": 10, "max": 300},
            {"id": "sample_every",  "label": "Sample every N (LK method)", "type": "number",
             "default": 1,  "min": 1,  "max": 10},
        ],
        "run": run_s2_object_tracker,
    },
    "s2_trajectory_extractor": {
        "id":    "s2_trajectory_extractor",
        "name":  "Trajectory Extractor",
        "desc":  "Convert tracks into position, velocity, acceleration, and contact-timing profiles, then flag statistically anomalous motion.",
        "badge": "medium",
        "dummy": False,
        "requires_pair": False,
        "settings": [
            {"id": "num_keypoints", "label": "Keypoints to track", "type": "number",
             "default": 60, "min": 10, "max": 300},
            {"id": "smoothing_window", "label": "Smoothing window (frames)", "type": "number",
             "default": 7, "min": 1, "max": 31},
            {"id": "z_threshold", "label": "Anomaly sensitivity (robust z)", "type": "number",
             "default": 3.5, "min": 1.5, "max": 8},
            {"id": "fps", "label": "Video FPS override (0 = auto)", "type": "number",
             "default": 0, "min": 0, "max": 240},
            {"id": "render_video", "label": "Kinematic overlay video", "type": "select",
             "default": "true",
             "options": [
                 {"value": "true",  "label": "Show annotated overlay"},
                 {"value": "false", "label": "Skip (faster)"},
             ]},
        ],
        "run": run_s2_trajectory_extractor,
    },
    "s2_event_localizer": {
        "id":    "s2_event_localizer",
        "name":  "Event Localizer",
        "desc":  "Crop the timeline into candidate failure windows from Stage 1 signals; emits a seek-able defect-marker viewer.",
        "badge": "medium",
        "dummy": False,
        "requires_pair": False,
        "settings": [
            {"id": "window_seconds", "label": "Merge signals within (seconds)", "type": "number",
             "default": 0.5, "min": 0.1, "max": 10},
            {"id": "min_signals_per_window", "label": "Min signals to form a marker", "type": "number",
             "default": 1, "min": 1, "max": 20},
            {"id": "min_test_severity", "label": "Min test severity to include (%)", "type": "number",
             "default": 15, "min": 0, "max": 100},
        ],
        "run": run_s2_event_localizer,
    },
    "s2_hypothesis_generator": {
        "id":    "s2_hypothesis_generator",
        "name":  "Physics Hypothesis Generator",
        "desc":  "Rank which Stage 3 specialists to run: heuristic priors from bus evidence + one VLM triage call over keyframes at flagged moments. (Absorbed the former Hypothesis Ranker.)",
        "badge": "medium",
        "dummy": False,
        "requires_pair": False,
        "settings": [
            _vlm_model_setting("Vision model"),
            _vlm_key_setting(),
            {"id": "max_hypotheses", "label": "Max hypotheses to return", "type": "number",
             "default": 4, "min": 1, "max": 8},
            {"id": "max_keyframes", "label": "Keyframes shown to the VLM", "type": "number",
             "default": 4, "min": 2, "max": 6},
        ],
        "run": run_s2_hypothesis_generator,
    },

    # ── Stage 3: Specialist Evaluation ───────────────────────────────────────
    "s3_consistency": {
        "id":    "s3_consistency",
        "name":  "Object Consistency Specialist",
        "desc":  "DINOv2 drift detects appearance change-points per masked subject (fixed viewport); the VLM verifies and explains each one. Vanish/reappear from mask presence gaps.",
        "badge": "expensive",
        "dummy": False,
        "requires_pair": False,
        "settings": [
            _vlm_model_setting("Vision model"),
            _vlm_key_setting(),
            {"id": "max_subjects", "label": "Max subjects (inline fallback)", "type": "number",
             "default": 3, "min": 1, "max": 6},
            {"id": "max_checks", "label": "Max VLM checks per subject", "type": "number",
             "default": 4, "min": 1, "max": 12},
            {"id": "drift_threshold", "label": "Drift threshold (cosine dist)", "type": "number",
             "default": 0.30, "min": 0.05, "max": 1.0},
            {"id": "strip_tiles", "label": "Tiles in overview strip", "type": "number",
             "default": 6, "min": 3, "max": 10},
            {"id": "min_vanish_gap_s", "label": "Min vanish gap (s)", "type": "number",
             "default": 0.3, "min": 0.05, "max": 3.0},
        ],
        "run": run_s3_consistency,
    },
    "s3_collision": {
        "id":    "s3_collision",
        "name":  "Collision & Contact Specialist",
        "desc":  "Mask-intersection contact episodes per subject pair: interpenetration (VLM-verified), restitution bounds (energy gain), and phantom bounces off nothing.",
        "badge": "expensive",
        "dummy": False,
        "requires_pair": False,
        "settings": [
            _vlm_model_setting("Vision model"),
            _vlm_key_setting(),
            {"id": "overlap_threshold", "label": "Contact threshold (dilated-mask adjacency)", "type": "number",
             "default": 0.02, "min": 0.005, "max": 0.9},
            {"id": "deep_overlap", "label": "Interpenetration threshold (raw mask overlap)", "type": "number",
             "default": 0.35, "min": 0.05, "max": 1.0},
            {"id": "restitution_max", "label": "Max plausible restitution", "type": "number",
             "default": 1.1, "min": 0.5, "max": 3.0},
            {"id": "max_checks", "label": "Max VLM confirmations", "type": "number",
             "default": 3, "min": 1, "max": 10},
            {"id": "max_subjects", "label": "Max subjects (inline fallback)", "type": "number",
             "default": 3, "min": 1, "max": 6},
        ],
        "run": run_s3_collision,
    },
    "s3_gravity": {
        "id":    "s3_gravity",
        "name":  "Gravity Specialist",
        "desc":  "Fit projectile trajectories and compare inferred gravitational acceleration to expected value.",
        "badge": "expensive",
        "dummy": True,
        "requires_pair": False,
        "settings": [
            {"id": "g_tolerance", "label": "Gravitational accel tolerance (fraction)", "type": "number",
             "default": 0.20, "min": 0.01, "max": 1.0},
        ],
        "run": run_s3_gravity,
    },
    "s3_momentum": {
        "id":    "s3_momentum",
        "name":  "Momentum Specialist",
        "desc":  "Check linear and angular momentum conservation across interaction events.",
        "badge": "expensive",
        "dummy": True,
        "requires_pair": False,
        "settings": [
            {"id": "momentum_tolerance", "label": "Momentum tolerance (fraction)", "type": "number",
             "default": 0.15, "min": 0.01, "max": 1.0},
            {"id": "energy_tolerance",   "label": "Energy tolerance (fraction)",   "type": "number",
             "default": 0.20, "min": 0.01, "max": 1.0},
        ],
        "run": run_s3_momentum,
    },
    "s3_friction": {
        "id":    "s3_friction",
        "name":  "Friction Specialist",
        "desc":  "Measure deceleration rates and infer friction coefficients; flag sliding anomalies.",
        "badge": "expensive",
        "dummy": True,
        "requires_pair": False,
        "settings": [
            {"id": "mu_min", "label": "Min plausible friction coefficient", "type": "number",
             "default": 0.0, "min": 0.0, "max": 5.0},
            {"id": "mu_max", "label": "Max plausible friction coefficient", "type": "number",
             "default": 1.5, "min": 0.0, "max": 5.0},
        ],
        "run": run_s3_friction,
    },
    "s3_deformation": {
        "id":    "s3_deformation",
        "name":  "Deformation Specialist",
        "desc":  "Analyse surface deformation of soft objects; check proportionality to applied force.",
        "badge": "expensive",
        "dummy": True,
        "requires_pair": False,
        "settings": [
            {"id": "deformation_threshold", "label": "Deformation threshold (normalised)", "type": "number",
             "default": 0.05, "min": 0.001, "max": 1.0},
        ],
        "run": run_s3_deformation,
    },
    # s3_contact merged into s3_collision — same events, same mask evidence.
    "s3_fluid": {
        "id":    "s3_fluid",
        "name":  "Fluid Specialist",
        "desc":  "Check fluid flow patterns for continuity, incompressibility, and viscosity violations.",
        "badge": "expensive",
        "dummy": True,
        "requires_pair": False,
        "settings": [
            {"id": "viscosity_mode", "label": "Viscosity regime", "type": "select",
             "default": "low",
             "options": [
                 {"value": "low",  "label": "Low viscosity (water)"},
                 {"value": "high", "label": "High viscosity (syrup)"},
             ]},
        ],
        "run": run_s3_fluid,
    },
    "s3_causality": {
        "id":    "s3_causality",
        "name":  "Causality Specialist",
        "desc":  "Verify effects follow causes with plausible delay; detect time-reversed sequences and temporal drift.",
        "badge": "expensive",
        "dummy": True,
        "requires_pair": False,
        "settings": [
            {"id": "max_causal_lag_frames", "label": "Max causal lag (frames)", "type": "number",
             "default": 3, "min": 1, "max": 30},
        ],
        "run": run_s3_causality,
    },

    # ── Stage 4: Final Diagnosis & Treatment Plan ─────────────────────────────
    "s4_scorer": {
        "id":    "s4_scorer",
        "name":  "Physics Consistency Scorer",
        "desc":  "Aggregate Stage 1–3 evidence into a single Physics Consistency Score (0–100).",
        "badge": "output",
        "dummy": True,
        "requires_pair": False,
        "settings": [
            {"id": "w_stage1", "label": "Stage 1 weight", "type": "number",
             "default": 0.2, "min": 0.0, "max": 1.0},
            {"id": "w_stage2", "label": "Stage 2 weight", "type": "number",
             "default": 0.3, "min": 0.0, "max": 1.0},
            {"id": "w_stage3", "label": "Stage 3 weight", "type": "number",
             "default": 0.5, "min": 0.0, "max": 1.0},
        ],
        "run": run_s4_scorer,
    },
    "s4_severity": {
        "id":    "s4_severity",
        "name":  "Severity Assessor",
        "desc":  "Map confirmed failure type and confidence onto Critical / Moderate / Minor / Inconclusive.",
        "badge": "output",
        "dummy": True,
        "requires_pair": False,
        "run": run_s4_severity,
    },
    "s4_pbt": {
        "id":    "s4_pbt",
        "name":  "Physics Breakdown Timer",
        "desc":  "Pinpoint the exact frame where the video first deviates from physical law (PBT).",
        "badge": "output",
        "dummy": True,
        "requires_pair": False,
        "settings": [
            {"id": "method", "label": "Change-point method", "type": "select",
             "default": "cusum",
             "options": [
                 {"value": "cusum", "label": "CUSUM"},
                 {"value": "bocpd", "label": "BOCPD"},
                 {"value": "pelt",  "label": "PELT"},
             ]},
        ],
        "run": run_s4_pbt,
    },
    "s4_explainer": {
        "id":    "s4_explainer",
        "name":  "Failure Explainer",
        "desc":  "Generate a structured, human-readable explanation of the detected physics failure.",
        "badge": "output",
        "dummy": True,
        "requires_pair": False,
        "settings": [
            {"id": "use_vlm", "label": "Enhance with VLM description", "type": "select",
             "default": "false",
             "options": [
                 {"value": "false", "label": "No"},
                 {"value": "true",  "label": "Yes (requires API key)"},
             ]},
        ],
        "run": run_s4_explainer,
    },
    "s4_report": {
        "id":    "s4_report",
        "name":  "Diagnostic Report",
        "desc":  "Compile all Stage 4 outputs into a final report: score, severity, PBT, explanation, and recommendations.",
        "badge": "output",
        "dummy": False,
        "requires_pair": False,
        "settings": [
            {"id": "output_format", "label": "Report format", "type": "select",
             "default": "json_visual",
             "options": [
                 {"value": "json_visual", "label": "JSON + Visual Summary"},
                 {"value": "json", "label": "JSON"},
                 {"value": "html", "label": "HTML"},
                 {"value": "pdf",  "label": "PDF"},
             ]},
             {"id": "use_llm_summary", "label": "Generate LLM summary", "type": "select",
              "default": "true",
              "options": [
                  {"value": "true", "label": "True"},
                  {"value": "false", "label": "False"},
             ]},
             _createai_key_setting(),
        ],
        "run": run_s4_report,
    },
}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="PhysicsLENS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Batch / dataset evaluation (browser upload + HuggingFace) — self-contained.
from dataset_api import build_dataset_router
app.include_router(build_dataset_router(PIPELINES))


@app.get("/pipelines")
def list_pipelines():
    return [
        {k: v for k, v in p.items() if k != "run"}
        for p in PIPELINES.values()
    ]


def _save_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    shutil.copyfileobj(upload.file, tmp)
    tmp.flush()
    return tmp.name


@app.post("/run")
async def run_pipeline(
    pipeline_id: str           = Form(...),
    video:       UploadFile    = File(...),
    video_ai:    Optional[UploadFile] = File(None),
    prompt:      Optional[str] = Form(None),
    settings:    Optional[str] = Form(None),   # JSON string of setting values
):
    if pipeline_id not in PIPELINES:
        return {"error": f"Unknown pipeline: {pipeline_id}"}

    p     = PIPELINES[pipeline_id]
    paths = []

    try:
        paths.append(_save_upload(video))
        if p.get("requires_pair"):
            if video_ai is None:
                return {"error": "This pipeline requires both a GT and AI video."}
            paths.append(_save_upload(video_ai))

        # Build kwargs for the pipeline run() function
        kwargs: dict = {}
        if p.get("requires_prompt"):
            kwargs["prompt"] = prompt
        if p.get("settings"):
            kwargs["settings"] = settings   # forwarded as raw JSON string

        pipeline_fn = p["run"]

        async def event_stream():
            try:
                if p.get("requires_pair"):
                    gen = pipeline_fn(paths[0], paths[1], **kwargs)
                else:
                    gen = pipeline_fn(paths[0], **kwargs)
                gen = instrument(gen, badge=p.get("badge", "—"))
                async for event in gen:
                    yield json.dumps(event) + "\n"
            except Exception as exc:
                yield json.dumps({"type": "error", "text": str(exc)}) + "\n"
            finally:
                for path in paths:
                    Path(path).unlink(missing_ok=True)

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    except Exception as exc:
        for path in paths:
            Path(path).unlink(missing_ok=True)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")

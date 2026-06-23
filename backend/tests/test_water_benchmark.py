import numpy as np
from pipelines.stage3.water_vbench_flow import analyze as vbench


def _smooth_pan(n=64, i=0):
    f = np.zeros((n, n, 3), dtype=np.uint8)
    f[:, :, 0] = np.clip((np.arange(n) + i * 2) % 255, 0, 255)[None, :]
    return f


def test_vbench_returns_single_score():
    frames = [_smooth_pan(i=i) for i in range(6)]
    res = vbench(frames, fps=30.0, cfg={"backend": "cpu"})
    assert "severity" in res and 0 <= res["severity"] <= 100
    assert "metrics" in res and isinstance(res["metrics"], list)


from pipelines.stage3.water_vlm_judge import parse_verdict, severity_from_verdicts


def test_parse_verdict_handles_fenced_json():
    raw = '```json\n{"plausibility": 0.2, "violations": ["water vanishes"], "explanation": "x"}\n```'
    v = parse_verdict(raw)
    assert v["plausibility"] == 0.2
    assert "water vanishes" in v["violations"]


def test_parse_verdict_is_robust_to_garbage():
    v = parse_verdict("not json at all")
    assert v["plausibility"] is None
    assert v["violations"] == []


def test_severity_from_verdicts_inverts_plausibility():
    vs = [{"plausibility": 1.0, "violations": [], "explanation": ""},
          {"plausibility": 0.0, "violations": ["x"], "explanation": ""}]
    s = severity_from_verdicts(vs)
    assert 40 <= s <= 60   # mean plausibility 0.5 → ~50
    assert severity_from_verdicts([]) == 0


from pipelines.stage4.water_benchmark import compare


def _blob(n=64, cx=20, r=10):
    f = np.zeros((n, n, 3), dtype=np.uint8)
    ys, xs = np.mgrid[0:n, 0:n]
    f[(xs - cx) ** 2 + (ys - n // 2) ** 2 <= r * r] = (200, 60, 20)
    return f


def test_compare_runs_all_homemade_methods_offline():
    frames = [_blob(cx=20 + 2 * i, r=6 + 3 * i) for i in range(6)]  # moving + growing
    out = compare(frames, fps=30.0, cfg={"backend": "cpu"})
    for key in ("incompressibility", "mass_conservation", "vorticity",
                "surface_coherence", "vbench_flow"):
        assert key in out["methods"]
        assert 0 <= out["methods"][key]["severity"] <= 100
    assert "vlm_judge" not in out["methods"]   # skipped without api_key
    assert isinstance(out["agreement"], list)

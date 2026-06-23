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

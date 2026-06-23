import numpy as np
from pipelines.stage3.water_incompressibility import analyze as incompressibility


def _moving_blob(n=64, cx=20, r=10):
    """A filled blue disk on black, centre cx, radius r."""
    f = np.zeros((n, n, 3), dtype=np.uint8)
    ys, xs = np.mgrid[0:n, 0:n]
    disk = (xs - cx) ** 2 + (ys - n // 2) ** 2 <= r * r
    f[disk] = (200, 60, 20)
    return f


def test_incompressibility_low_for_rigid_translation():
    # disk translates right by 3px/frame: incompressible, divergence ~0
    frames = [_moving_blob(cx=20 + 3 * i) for i in range(6)]
    res = incompressibility(frames, fps=30.0, cfg={"backend": "cpu"})
    assert res["severity"] < 40
    assert set(res) >= {"time", "series", "severity", "signals", "metrics", "color"}


def test_incompressibility_high_for_expanding_blob():
    # disk grows each frame: strong positive divergence (mass creation)
    frames = [_moving_blob(r=6 + 4 * i) for i in range(6)]
    res = incompressibility(frames, fps=30.0, cfg={"backend": "cpu"})
    assert res["severity"] > 25
    assert len(res["signals"]) >= 1

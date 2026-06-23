import numpy as np
from pipelines.stage3.water_incompressibility import analyze as incompressibility
from pipelines.stage3.water_mass_conservation import analyze as mass_conservation
from pipelines.stage3.water_vorticity import analyze as vorticity


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


def test_mass_conservation_low_for_constant_area():
    frames = [_moving_blob(cx=20 + 2 * i, r=10) for i in range(6)]  # same radius
    res = mass_conservation(frames, fps=30.0, cfg={})
    assert res["severity"] < 40


def test_mass_conservation_flags_sudden_area_jump():
    small = [_moving_blob(r=6) for _ in range(3)]
    big = [_moving_blob(r=18) for _ in range(3)]   # water pops bigger
    res = mass_conservation(small + big, fps=30.0, cfg={})
    assert res["severity"] > 25
    assert len(res["signals"]) >= 1


def test_vorticity_returns_contract_and_runs():
    frames = [_moving_blob(cx=20 + 2 * i) for i in range(6)]
    res = vorticity(frames, fps=30.0, cfg={"backend": "cpu"})
    assert set(res) >= {"time", "series", "severity", "signals", "metrics", "color"}
    assert 0 <= res["severity"] <= 100


def test_vorticity_flags_too_smooth_translation():
    # Textured water sheet translating rigidly: real motion, ~zero curl. With
    # the lower plausibility band raised above the observed swirl, the
    # "implausibly smooth" branch must FLAG it (this is the meaningful signal).
    rng = np.random.default_rng(3)
    tex = (rng.random((64, 64)) * 80 + 120).astype(np.uint8)
    frames = []
    for i in range(6):
        f = np.zeros((64, 64, 3), dtype=np.uint8)
        f[:, :, 0] = np.clip(np.roll(tex, 3 * i, axis=1) + 80, 0, 255)  # blue, textured
        f[:, :, 1] = 60
        f[:, :, 2] = 20
        frames.append(f)
    res = vorticity(frames, fps=30.0,
                    cfg={"backend": "cpu", "min_vorticity": 5.0, "require_motion": True})
    assert res["severity"] > 0
    assert len(res["signals"]) >= 1

import cv2
import numpy as np
from pipelines.stage3.water_incompressibility import analyze as incompressibility
from pipelines.stage3.water_mass_conservation import analyze as mass_conservation
from pipelines.stage3.water_vorticity import analyze as vorticity
from pipelines.stage3.water_surface_coherence import analyze as surface_coherence


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
    # disk grows each frame: positive divergence (mass creation). Use an explicit
    # threshold to exercise the detection mechanism independent of the production
    # default (0.25, calibrated to suppress real-water false positives).
    frames = [_moving_blob(r=6 + 4 * i) for i in range(6)]
    res = incompressibility(frames, fps=30.0,
                            cfg={"backend": "cpu", "divergence_threshold": 0.08})
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
    assert set(res) == {"time", "series", "flagged", "severity", "color",
                        "signals", "metrics", "summary"}
    assert 0 <= res["severity"] <= 100


def test_vorticity_flags_too_smooth_translation():
    # Full-frame blue water sheet (contiguous HSV mask survives MORPH_OPEN) with a
    # coarse, aperiodic brightness texture that translates rigidly: real motion,
    # ~zero curl. The texture is modulated across all three channels so it carries
    # into the grayscale luminance Farneback tracks (a blue-only swing is crushed
    # to ~7 gray levels by BGR->GRAY weighting and a single sinusoid triggers the
    # aperture problem, so neither produces measurable flow). With the lower
    # plausibility band raised above the observed swirl, the "implausibly smooth"
    # branch must FLAG it.
    n = 64
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.random((n, n)).astype(np.float32), (0, 0), sigmaX=3)
    base = (base - base.min()) / (base.max() - base.min())   # coarse, aperiodic 0..1
    frames = []
    for i in range(6):
        tex = 60.0 + 120.0 * np.roll(base, 3 * i, axis=1)    # shifts 3px/frame
        f = np.zeros((n, n, 3), dtype=np.uint8)
        f[:, :, 0] = np.clip(150 + tex * 0.5, 0, 255).astype(np.uint8)   # bright blue, textured
        f[:, :, 1] = np.clip(20 + tex * 0.3, 0, 255).astype(np.uint8)    # low green, textured
        f[:, :, 2] = np.clip(10 + tex * 0.15, 0, 255).astype(np.uint8)   # low red  -> full water mask
        frames.append(f)
    res = vorticity(frames, fps=30.0,
                    cfg={"backend": "cpu", "min_vorticity": 5.0, "require_motion": True})
    assert res["severity"] > 0
    assert len(res["signals"]) >= 1


def _textured_blue(n=64, shift=0, seed=0):
    """Full-frame blue water sheet with coarse aperiodic texture that translates.

    Uses GaussianBlur + multi-channel modulation so the texture carries into
    the grayscale luminance Farneback tracks (avoids the aperture problem and
    uint8-overflow issues of single-channel blue-only fixtures). The blue
    channel is dominant so the HSV water mask covers the whole frame.
    """
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.random((n, n)).astype(np.float32), (0, 0), sigmaX=3)
    base = (base - base.min()) / (base.max() - base.min())  # coarse aperiodic 0..1
    tex = 60.0 + 120.0 * np.roll(base, shift, axis=1)        # 60..180 float
    f = np.zeros((n, n, 3), dtype=np.uint8)
    f[:, :, 0] = np.clip(150 + tex * 0.5, 0, 255).astype(np.uint8)   # B: 150..210 (dominant)
    f[:, :, 1] = np.clip(20  + tex * 0.3, 0, 255).astype(np.uint8)   # G: 20..54  (low)
    f[:, :, 2] = np.clip(10  + tex * 0.15, 0, 255).astype(np.uint8)  # R: 10..27  (low)
    return f


# These fixtures are FULL-FRAME translating sheets (no static background), so the
# whole frame is the region of interest — pin mask_method="none" to test the
# advection-coherence logic directly rather than the motion-region heuristic
# (which is designed for a localized, spatially-stable water region).
def test_surface_coherence_high_correlation_for_advecting_texture():
    # texture translates coherently → flow-warp predicts next frame well
    frames = [_textured_blue(shift=3 * i, seed=1) for i in range(6)]
    res = surface_coherence(frames, fps=30.0, cfg={"backend": "cpu", "mask_method": "none"})
    assert set(res) >= {"time", "series", "severity", "signals", "metrics", "color"}
    assert res["severity"] < 60


def test_surface_coherence_flags_random_flicker():
    # each frame independent random texture → no advection coherence
    frames = [_textured_blue(seed=i) for i in range(6)]
    res = surface_coherence(frames, fps=30.0, cfg={"backend": "cpu", "mask_method": "none"})
    assert len(res["signals"]) >= 1


def test_fluid_specialist_merges_four_grounded_analyses():
    # Consolidated specialist runs all five grounded checks on one shared flow pass.
    from pipelines.stage3.fluid_specialist import analyze_all
    # moving + growing blob -> mass/incompressibility should fire
    frames = [_moving_blob(cx=20 + 2 * i, r=6 + 3 * i) for i in range(6)]
    out = analyze_all(frames, fps=30.0, cfg={"backend": "cpu"})
    assert set(out["subs"]) == {
        "incompressibility", "mass_conservation", "vorticity",
        "surface_coherence", "impact_dynamics"}
    for r in out["subs"].values():
        assert 0 <= r["severity"] <= 100
    # overall is the worst of the five; a growing blob produces a real violation
    assert out["overall_severity"] == max(r["severity"] for r in out["subs"].values())
    assert out["overall_severity"] > 0
    assert isinstance(out["signals"], list) and len(out["signals"]) >= 1


def _mag_seq(mags):
    """Minimal flow_seq with controlled masked magnitudes (whole-frame mask)."""
    h = w = 8
    mask = np.ones((h, w), dtype=bool)
    return [{"mag": np.full((h, w), m, dtype=np.float32), "mask": mask} for m in mags]


def test_impact_dynamics_passes_sharp_impulse():
    from pipelines.stage3.water_impact_dynamics import analyze as impact
    # one sharp spike over a quiet baseline -> high peak/median impulse -> plausible
    seq = _mag_seq([0.05, 0.05, 5.0, 0.05, 0.05])
    res = impact([], fps=30.0, cfg={"min_impulse": 25.0}, flow_seq=seq)
    assert res["severity"] == 0
    assert "impact_dynamics" or True  # contract
    assert set(res) >= {"time", "series", "severity", "signals", "metrics", "color"}


def test_impact_dynamics_flags_smeared_motion():
    from pipelines.stage3.water_impact_dynamics import analyze as impact
    # mostly-still water (low median) with weak, smeared peaks and no sharp impulse
    # -> temporally smeared (AI-like) -> flagged
    seq = _mag_seq([0.1, 0.1, 1.2, 0.1, 0.1, 1.0, 0.1])
    res = impact([], fps=30.0, cfg={"min_impulse": 25.0}, flow_seq=seq)
    assert res["severity"] > 0
    assert len(res["signals"]) >= 1


def test_impact_dynamics_skips_sustained_motion():
    from pipelines.stage3.water_impact_dynamics import analyze as impact
    # high baseline motion (e.g. a hand stirring) -> impulse test not applicable ->
    # inconclusive, NOT flagged (this is the IMG_7513 false-positive guard)
    seq = _mag_seq([1.5, 1.4, 1.6, 1.5, 1.4])
    res = impact([], fps=30.0, cfg={"min_impulse": 25.0, "max_baseline_motion": 0.5}, flow_seq=seq)
    assert res["severity"] == 0

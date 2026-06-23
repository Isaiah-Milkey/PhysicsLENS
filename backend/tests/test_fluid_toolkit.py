import numpy as np
from tools.fluid import helmholtz, flow_magnitude, masked_mean, severity_color


def _grid(n=40):
    ys, xs = np.mgrid[0:n, 0:n]
    return xs.astype(np.float32), ys.astype(np.float32)


def _interior(a, p=4):
    return a[p:-p, p:-p]


def test_translation_field_is_divergence_and_curl_free():
    x, y = _grid()
    u = np.full_like(x, 3.0)
    v = np.full_like(y, -2.0)
    div, curl = helmholtz(u, v)
    assert abs(_interior(div).mean()) < 1e-4
    assert abs(_interior(curl).mean()) < 1e-4


def test_radial_source_has_positive_divergence():
    x, y = _grid()
    div, curl = helmholtz(x, y)          # u=x, v=y
    assert _interior(div).mean() > 1.5   # analytic value 2.0
    assert abs(_interior(curl).mean()) < 1e-4


def test_rigid_rotation_has_curl_no_divergence():
    x, y = _grid()
    div, curl = helmholtz(-y, x)         # u=-y, v=x
    assert abs(_interior(div).mean()) < 1e-4
    assert _interior(curl).mean() > 1.5  # analytic value 2.0


def test_flow_magnitude_and_masked_mean():
    u = np.array([[3.0, 0.0]], dtype=np.float32)
    v = np.array([[4.0, 0.0]], dtype=np.float32)
    mag = flow_magnitude(u, v)
    assert np.allclose(mag, [[5.0, 0.0]])
    mask = np.array([[True, False]])
    assert abs(masked_mean(mag, mask) - 5.0) < 1e-6
    assert masked_mean(mag, np.zeros_like(mask)) == 0.0


def test_severity_color_bands():
    assert severity_color(50) == "#E24B4A"
    assert severity_color(20) == "#EF9F27"
    assert severity_color(5) == "#4CAF50"

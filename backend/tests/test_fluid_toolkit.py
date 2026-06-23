import cv2
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


from tools.fluid import dense_flow


def test_dense_flow_recovers_uniform_translation():
    rng = np.random.default_rng(0)
    base = (rng.random((64, 64)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 2.0)
    shifted = np.roll(base, shift=(0, 3), axis=(0, 1))  # +3 px in x
    u, v = dense_flow(base, shifted, backend="cpu")
    assert u.shape == base.shape
    c = slice(8, -8)
    assert abs(u[c, c].mean() - 3.0) < 1.0
    assert abs(v[c, c].mean()) < 1.0


def test_dense_flow_auto_returns_valid_shape():
    img = np.zeros((32, 32), dtype=np.uint8)
    u, v = dense_flow(img, img, backend="auto")
    assert u.shape == (32, 32) and v.shape == (32, 32)

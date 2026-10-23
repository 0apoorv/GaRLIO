import numpy as np
import pytest

from garlio.so3 import Exp, Log, hat, vee


def random_rotation(rng):
    phi = rng.normal(size=3)
    phi = phi / np.linalg.norm(phi) * rng.uniform(0.01, np.pi - 0.01)
    return Exp(phi)


def test_hat_vee_roundtrip():
    rng = np.random.default_rng(0)
    v = rng.normal(size=3)
    assert np.allclose(vee(hat(v)), v)


def test_hat_is_skew_symmetric():
    v = np.array([1.0, 2.0, 3.0])
    H = hat(v)
    assert np.allclose(H, -H.T)


def test_exp_zero_is_identity():
    assert np.allclose(Exp(np.zeros(3)), np.eye(3))


def test_log_identity_is_zero():
    assert np.allclose(Log(np.eye(3)), np.zeros(3))


@pytest.mark.parametrize("seed", range(10))
def test_exp_log_roundtrip(seed):
    rng = np.random.default_rng(seed)
    phi = rng.normal(size=3)
    phi = phi / np.linalg.norm(phi) * rng.uniform(0.01, np.pi - 0.01)
    R = Exp(phi)
    phi_back = Log(R)
    assert np.allclose(phi, phi_back, atol=1e-8)


def test_exp_produces_orthonormal_rotation():
    rng = np.random.default_rng(1)
    for _ in range(20):
        R = random_rotation(rng)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_exp_small_angle_matches_first_order():
    phi = np.array([1e-5, -2e-5, 3e-5])
    R = Exp(phi)
    R_first_order = np.eye(3) + hat(phi)
    assert np.allclose(R, R_first_order, atol=1e-9)


def test_exp_known_90deg_about_z():
    phi = np.array([0.0, 0.0, np.pi / 2])
    R = Exp(phi)
    expected = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert np.allclose(R, expected, atol=1e-10)

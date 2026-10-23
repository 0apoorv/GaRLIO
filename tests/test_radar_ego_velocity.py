import numpy as np
import pytest

from garlio.radar import estimate_ego_velocity_ransac


def synthetic_static_points(n, v_ego, rng, doppler_noise=0.0):
    points = rng.uniform(-10, 10, size=(n, 3))
    points[:, 2] = rng.uniform(-1, 1, size=n)
    u = points / np.linalg.norm(points, axis=1, keepdims=True)
    doppler = -u @ v_ego
    if doppler_noise > 0:
        doppler = doppler + rng.normal(scale=doppler_noise, size=n)
    return points, doppler


def test_recovers_exact_velocity_from_three_noiseless_points():
    rng = np.random.default_rng(0)
    v_true = np.array([2.0, -1.0, 0.3])
    points, doppler = synthetic_static_points(3, v_true, rng)

    result = estimate_ego_velocity_ransac(points, doppler, inlier_thresh=1e-6, rng=rng)

    assert np.allclose(result.v_ego, v_true, atol=1e-6)
    assert result.static_mask.all()


def test_recovers_velocity_from_many_noisy_static_points():
    rng = np.random.default_rng(1)
    v_true = np.array([3.0, 0.5, -0.2])
    points, doppler = synthetic_static_points(100, v_true, rng, doppler_noise=0.01)

    result = estimate_ego_velocity_ransac(points, doppler, inlier_thresh=0.1, rng=rng)

    assert np.allclose(result.v_ego, v_true, atol=0.05)
    assert result.static_mask.sum() > 90


def test_rejects_dynamic_outliers_and_still_recovers_true_velocity():
    rng = np.random.default_rng(2)
    v_true = np.array([5.0, -2.0, 0.0])
    n_static, n_dynamic = 80, 20

    static_points, static_doppler = synthetic_static_points(
        n_static, v_true, rng, doppler_noise=0.01
    )
    dynamic_points = rng.uniform(-10, 10, size=(n_dynamic, 3))
    dynamic_points[:, 2] = rng.uniform(-1, 1, size=n_dynamic)
    dynamic_doppler = rng.uniform(-8, 8, size=n_dynamic)

    points = np.vstack([static_points, dynamic_points])
    doppler = np.concatenate([static_doppler, dynamic_doppler])
    true_static_mask = np.concatenate(
        [np.ones(n_static, dtype=bool), np.zeros(n_dynamic, dtype=bool)]
    )

    result = estimate_ego_velocity_ransac(
        points, doppler, inlier_thresh=0.15, n_iters=300, rng=rng
    )

    assert np.allclose(result.v_ego, v_true, atol=0.1)
    agreement = (result.static_mask == true_static_mask).mean()
    assert agreement > 0.9


def test_dynamic_and_static_masks_are_complementary():
    rng = np.random.default_rng(3)
    v_true = np.array([1.0, 1.0, 1.0])
    points, doppler = synthetic_static_points(50, v_true, rng, doppler_noise=0.02)

    result = estimate_ego_velocity_ransac(points, doppler, rng=rng)

    assert np.array_equal(result.dynamic_mask, ~result.static_mask)


def test_raises_on_too_few_points():
    rng = np.random.default_rng(4)
    points = rng.uniform(-1, 1, size=(2, 3))
    doppler = rng.uniform(-1, 1, size=2)
    with pytest.raises(ValueError):
        estimate_ego_velocity_ransac(points, doppler, rng=rng)


def test_zero_ego_velocity_gives_near_zero_doppler():
    rng = np.random.default_rng(5)
    v_true = np.zeros(3)
    points, doppler = synthetic_static_points(30, v_true, rng, doppler_noise=0.01)

    result = estimate_ego_velocity_ransac(points, doppler, rng=rng)

    assert np.allclose(result.v_ego, np.zeros(3), atol=0.05)

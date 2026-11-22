import numpy as np

from garlio.evaluation import align_umeyama, compute_ate, compute_relative_errors
from garlio.so3 import Exp


def test_align_umeyama_recovers_known_rigid_transform():
    rng = np.random.default_rng(0)
    source = rng.uniform(-10, 10, size=(50, 3))

    true_R = Exp(np.array([0.1, -0.2, 0.3]))
    true_t = np.array([5.0, -3.0, 1.0])
    target = (true_R @ source.T).T + true_t

    aligned, R, t = align_umeyama(source, target)

    assert np.allclose(aligned, target, atol=1e-8)
    assert np.allclose(R, true_R, atol=1e-8)
    assert np.allclose(t, true_t, atol=1e-8)


def test_align_umeyama_with_noise_gives_small_residual():
    rng = np.random.default_rng(1)
    source = rng.uniform(-10, 10, size=(200, 3))

    true_R = Exp(np.array([0.0, 0.0, 0.5]))
    true_t = np.array([2.0, 1.0, 0.0])
    target = (true_R @ source.T).T + true_t
    target += rng.normal(0, 0.01, size=target.shape)

    aligned, R, t = align_umeyama(source, target)
    errors = np.linalg.norm(aligned - target, axis=1)

    assert np.mean(errors) < 0.02


def test_compute_ate_perfect_match_gives_zero_error():
    rng = np.random.default_rng(2)
    times = np.arange(0, 10, 0.1)
    est_pos = np.column_stack([np.sin(times), np.cos(times), times * 0.1])

    true_R = Exp(np.array([0.2, 0.0, 0.0]))
    true_t = np.array([100.0, 50.0, -20.0])
    gt_pos = (true_R @ est_pos.T).T + true_t

    result = compute_ate(times, est_pos, times, gt_pos)
    assert result["rmse"] < 1e-8
    assert result["n_matched"] == len(times)


def test_compute_ate_detects_real_drift():
    times = np.arange(0, 10, 0.1)
    gt_pos = np.column_stack([times, np.zeros_like(times), np.zeros_like(times)])
    # estimated trajectory curves away quadratically (simulating unbounded dead-reckoning
    # drift) -- curvature can't be explained away by a single global rigid rotation+translation,
    # unlike a straight-line shear, so the residual after best-fit alignment stays large.
    est_pos = gt_pos.copy()
    est_pos[:, 1] += 0.05 * times ** 2

    result = compute_ate(times, est_pos, times, gt_pos)
    assert result["rmse"] > 0.3


def test_match_by_time_respects_tolerance():
    from garlio.evaluation import match_by_time

    est_times = np.array([0.0, 1.0, 2.0])
    est_pos = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    gt_times = np.array([0.01, 5.0, 2.02])
    gt_pos = np.array([[0.0, 1, 0], [9, 9, 9], [2.0, 1, 0]])

    m_est, m_gt, m_t = match_by_time(est_times, est_pos, gt_times, gt_pos, max_dt=0.05)
    assert len(m_est) == 2
    assert np.allclose(m_t, [0.0, 2.0])


def test_relative_errors_are_zero_for_identical_poses():
    times = np.arange(4.0)
    pos = np.column_stack([times, times ** 2, np.zeros_like(times)])
    rotations = np.array([Exp(np.array([0.0, 0.0, 0.1 * t])) for t in times])

    result = compute_relative_errors(times, pos, rotations, times, pos, rotations, interval_s=1.0)

    assert result["n_pairs"] == 3
    assert result["translation_rmse"] < 1e-12
    assert result["rotation_rmse_rad"] < 1e-8


def test_relative_errors_ignore_a_shared_global_rigid_transform():
    times = np.arange(4.0)
    pos = np.column_stack([times, np.sin(times), np.zeros_like(times)])
    rotations = np.array([Exp(np.array([0.0, 0.0, 0.1 * t])) for t in times])
    global_R = Exp(np.array([0.2, -0.1, 0.3]))
    global_t = np.array([5.0, -3.0, 2.0])
    transformed_pos = (global_R @ pos.T).T + global_t
    transformed_rotations = np.array([global_R @ R for R in rotations])

    result = compute_relative_errors(
        times, transformed_pos, transformed_rotations,
        times, pos, rotations, interval_s=1.0,
    )

    assert result["translation_rmse"] < 1e-12
    assert result["rotation_rmse_rad"] < 1e-8


def test_relative_errors_report_known_translation_and_rotation_error():
    times = np.arange(4.0)
    gt_pos = np.zeros((len(times), 3))
    est_pos = np.column_stack([np.zeros_like(times), 0.5 * times, np.zeros_like(times)])
    gt_rotations = np.repeat(np.eye(3)[None, :, :], len(times), axis=0)
    angle_per_second = np.deg2rad(10.0)
    est_rotations = np.array([
        Exp(np.array([0.0, 0.0, angle_per_second * t])) for t in times
    ])

    result = compute_relative_errors(
        times, est_pos, est_rotations,
        times, gt_pos, gt_rotations, interval_s=1.0,
    )

    assert np.allclose(result["translation_errors"], 0.5)
    assert np.allclose(result["rotation_errors_deg"], 10.0)


def test_relative_errors_handle_irregular_timestamps_with_duration_tolerance():
    times = np.array([0.0, 0.9, 2.1, 3.0])
    pos = np.column_stack([times, np.zeros_like(times), np.zeros_like(times)])
    rotations = np.repeat(np.eye(3)[None, :, :], len(times), axis=0)

    result = compute_relative_errors(
        times, pos, rotations, times, pos, rotations,
        interval_s=1.0, interval_tolerance_s=0.21,
    )

    assert result["n_pairs"] == 3
    assert np.allclose(result["pair_durations"], [0.9, 1.2, 0.9])


def test_relative_errors_raise_when_no_interval_pairs_are_available():
    times = np.array([0.0, 0.1])
    pos = np.zeros((2, 3))
    rotations = np.repeat(np.eye(3)[None, :, :], 2, axis=0)

    import pytest

    with pytest.raises(ValueError, match="No valid pose pairs"):
        compute_relative_errors(
            times, pos, rotations, times, pos, rotations,
            interval_s=1.0,
        )

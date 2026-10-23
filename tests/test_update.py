import numpy as np

from garlio.eskf import N_ERR
from garlio.so3 import Exp
from garlio.state import ImuState
from garlio.update import (
    ekf_update,
    iterated_ekf_update,
    point_to_plane_residual,
    radar_doppler_residual,
    stack_residuals,
)


def floor_points_in_body_frame(true_state, rng, n=40):
    world_pts = np.column_stack(
        [rng.uniform(-5, 5, n), rng.uniform(-5, 5, n), np.zeros(n)]
    )
    body_pts = true_state.R.T @ (world_pts - true_state.p).T
    return body_pts.T


def test_point_to_plane_residual_zero_at_true_state():
    true_state = ImuState()
    r, H = point_to_plane_residual(
        true_state, p_lidar=np.array([1.0, 2.0, 0.0]), normal=np.array([0.0, 0.0, 1.0]),
        centroid=np.zeros(3),
    )
    assert abs(r) < 1e-12
    assert H.shape == (N_ERR,)


def test_lidar_only_update_corrects_height_and_tilt_error():
    rng = np.random.default_rng(0)
    true_state = ImuState()
    body_pts = floor_points_in_body_frame(true_state, rng, n=60)

    biased_state = ImuState(
        R=Exp(np.array([0.05, -0.03, 0.0])),
        p=np.array([0.1, -0.05, 0.6]),
    )
    P0 = np.eye(N_ERR) * 0.5

    def residual_fn(state):
        entries = [
            point_to_plane_residual(state, p, np.array([0.0, 0.0, 1.0]), np.zeros(3))
            + (0.01,)
            for p in body_pts
        ]
        return stack_residuals(entries)

    new_state, new_P = iterated_ekf_update(biased_state, P0, residual_fn, n_iters=5)

    err_before = abs(biased_state.p[2])
    err_after = abs(new_state.p[2])
    assert err_after < err_before * 0.2

    tilt_before = np.linalg.norm(biased_state.R - true_state.R)
    tilt_after = np.linalg.norm(new_state.R - true_state.R)
    assert tilt_after < tilt_before * 0.3


def test_radar_only_update_corrects_velocity():
    rng = np.random.default_rng(1)
    v_true = np.array([4.0, -1.0, 0.2])
    true_state = ImuState(v=v_true)

    n = 30
    radar_pts = rng.uniform(-1, 1, size=(n, 3))
    radar_pts[:, 2] = rng.uniform(-0.3, 0.3, n)
    u = radar_pts / np.linalg.norm(radar_pts, axis=1, keepdims=True)
    v_body_true = true_state.R.T @ v_true
    doppler = -u @ v_body_true

    biased_state = ImuState(v=v_true + np.array([0.8, -0.6, 0.3]))
    P0 = np.eye(N_ERR) * 0.5
    P0[0:3, 0:3] = np.eye(3) * 1e-6

    def residual_fn(state):
        entries = [
            radar_doppler_residual(state, radar_pts[i], doppler[i]) + (0.01,)
            for i in range(n)
        ]
        return stack_residuals(entries)

    new_state, new_P = iterated_ekf_update(biased_state, P0, residual_fn, n_iters=5)

    err_before = np.linalg.norm(biased_state.v - v_true)
    err_after = np.linalg.norm(new_state.v - v_true)
    assert err_after < err_before * 0.2


def room_points_in_body_frame(true_state, rng, n_per_surface=30):
    floor = np.column_stack(
        [rng.uniform(-5, 5, n_per_surface), rng.uniform(-5, 5, n_per_surface), np.zeros(n_per_surface)]
    )
    wall_x = np.column_stack(
        [np.full(n_per_surface, 5.0), rng.uniform(-5, 5, n_per_surface), rng.uniform(0, 3, n_per_surface)]
    )
    wall_y = np.column_stack(
        [rng.uniform(-5, 5, n_per_surface), np.full(n_per_surface, 5.0), rng.uniform(0, 3, n_per_surface)]
    )
    surfaces = [
        (floor, np.array([0.0, 0.0, 1.0]), np.zeros(3)),
        (wall_x, np.array([1.0, 0.0, 0.0]), np.array([5.0, 0.0, 1.5])),
        (wall_y, np.array([0.0, 1.0, 0.0]), np.array([0.0, 5.0, 1.5])),
    ]
    entries = []
    for world_pts, normal, centroid in surfaces:
        body_pts = (true_state.R.T @ (world_pts - true_state.p).T).T
        entries.append((body_pts, normal, centroid))
    return entries


def test_combined_update_reduces_error_more_than_either_alone():
    rng = np.random.default_rng(2)
    true_state = ImuState(v=np.array([2.0, 1.0, 0.0]))
    room_surfaces = room_points_in_body_frame(true_state, rng, n_per_surface=30)

    n_radar = 25
    radar_pts = rng.uniform(-1, 1, size=(n_radar, 3))
    radar_pts[:, 2] = rng.uniform(-0.3, 0.3, n_radar)
    u = radar_pts / np.linalg.norm(radar_pts, axis=1, keepdims=True)
    v_body_true = true_state.R.T @ true_state.v
    doppler = -u @ v_body_true

    biased_state = ImuState(
        R=Exp(np.array([0.04, 0.02, 0.0])),
        v=true_state.v + np.array([0.5, -0.4, 0.1]),
        p=np.array([0.05, 0.05, 0.4]),
    )
    P0 = np.eye(N_ERR) * 0.5

    def residual_fn(state):
        lidar_entries = [
            point_to_plane_residual(state, p, normal, centroid) + (0.01,)
            for body_pts, normal, centroid in room_surfaces
            for p in body_pts
        ]
        radar_entries = [
            radar_doppler_residual(state, radar_pts[i], doppler[i]) + (0.02,)
            for i in range(n_radar)
        ]
        return stack_residuals(lidar_entries + radar_entries)

    new_state, new_P = iterated_ekf_update(biased_state, P0, residual_fn, n_iters=6)

    pos_err_before = np.linalg.norm(biased_state.p - true_state.p)
    pos_err_after = np.linalg.norm(new_state.p - true_state.p)
    vel_err_before = np.linalg.norm(biased_state.v - true_state.v)
    vel_err_after = np.linalg.norm(new_state.v - true_state.v)

    assert pos_err_after < pos_err_before * 0.2
    assert vel_err_after < vel_err_before * 0.2


def test_updated_covariance_is_symmetric_psd_and_shrinks():
    rng = np.random.default_rng(3)
    true_state = ImuState()
    body_pts = floor_points_in_body_frame(true_state, rng, n=50)

    biased_state = ImuState(p=np.array([0.0, 0.0, 0.3]))
    P0 = np.eye(N_ERR) * 0.5

    def residual_fn(state):
        entries = [
            point_to_plane_residual(state, p, np.array([0.0, 0.0, 1.0]), np.zeros(3))
            + (0.01,)
            for p in body_pts
        ]
        return stack_residuals(entries)

    new_state, new_P = iterated_ekf_update(biased_state, P0, residual_fn, n_iters=4)

    assert np.allclose(new_P, new_P.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(new_P)
    assert np.all(eigvals >= -1e-9)
    assert np.trace(new_P) < np.trace(P0)


def test_single_shot_ekf_update_matches_manual_kalman_gain():
    state = ImuState(p=np.array([0.0, 0.0, 1.0]))
    P = np.eye(N_ERR) * 1.0

    r, H = point_to_plane_residual(state, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.zeros(3))
    r_vec = np.array([r])
    H_mat = H.reshape(1, N_ERR)
    var = np.array([0.05])

    new_state, new_P = ekf_update(state, P, r_vec, H_mat, var)

    S = H_mat @ P @ H_mat.T + np.diag(var)
    K = P @ H_mat.T @ np.linalg.inv(S)
    expected_dx = (-K @ r_vec).flatten()
    expected_state = state.boxplus(expected_dx)

    assert np.allclose(new_state.p, expected_state.p, atol=1e-10)
    assert np.allclose(new_state.R, expected_state.R, atol=1e-10)

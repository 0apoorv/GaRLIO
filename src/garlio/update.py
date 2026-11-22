from typing import Callable

import numpy as np

from .so3 import hat
from .state import ImuState

N_ERR = 15


def point_to_plane_residual(
    state: ImuState,
    p_lidar: np.ndarray,
    normal: np.ndarray,
    centroid: np.ndarray,
    T_imu_lidar: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    p_lidar = np.asarray(p_lidar, dtype=float).reshape(3)
    normal = np.asarray(normal, dtype=float).reshape(3)
    centroid = np.asarray(centroid, dtype=float).reshape(3)

    if T_imu_lidar is None:
        T_imu_lidar = np.eye(4)
    T_imu_lidar = np.asarray(T_imu_lidar, dtype=float)
    p_imu = T_imu_lidar[:3, :3] @ p_lidar + T_imu_lidar[:3, 3]
    p_global = state.R @ p_imu + state.p
    r = float(normal @ (p_global - centroid))

    H = np.zeros(N_ERR)
    H[0:3] = -(normal @ state.R @ hat(p_imu))
    H[6:9] = normal
    return r, H


def radar_doppler_residual(
    state: ImuState,
    p_radar: np.ndarray,
    doppler_measured: float,
    T_imu_radar: np.ndarray | None = None,
    angular_velocity_imu: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    p_radar = np.asarray(p_radar, dtype=float).reshape(3)
    u = p_radar / np.linalg.norm(p_radar)

    if T_imu_radar is None:
        T_imu_radar = np.eye(4)
    T_imu_radar = np.asarray(T_imu_radar, dtype=float)
    R_imu_radar = T_imu_radar[:3, :3]
    t_imu_radar = T_imu_radar[:3, 3]

    v_imu = state.R.T @ state.v
    omega_imu = np.zeros(3) if angular_velocity_imu is None else np.asarray(angular_velocity_imu, dtype=float)
    v_imu_radar = v_imu + np.cross(omega_imu - state.bg, t_imu_radar)
    v_radar = R_imu_radar.T @ v_imu_radar
    predicted = -float(u @ v_radar)
    r = predicted - float(doppler_measured)

    H = np.zeros(N_ERR)
    H[0:3] = -(u @ R_imu_radar.T @ hat(v_imu))
    H[3:6] = -(u @ R_imu_radar.T @ state.R.T)
    H[9:12] = -(u @ R_imu_radar.T @ _hat(t_imu_radar))
    return r, H


def gravity_direction_residual(
    state: ImuState,
    previous_state: ImuState,
    delta_v_body: np.ndarray,
    gravity_reference: np.ndarray,
    dt: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Gravity-direction residual from IMU preintegration and endpoint velocities.

    ``delta_v_body`` is the bias-corrected specific-force integral expressed in
    the previous IMU frame. The residual is a scalar cosine error on S2.
    """
    if dt <= 0:
        raise ValueError("gravity interval must be positive")
    delta_v_body = np.asarray(delta_v_body, dtype=float).reshape(3)
    g_ref = np.asarray(gravity_reference, dtype=float).reshape(3).copy()
    g_ref /= np.linalg.norm(g_ref)
    g_est = (state.v - previous_state.v - previous_state.R @ delta_v_body) / dt
    g_norm = np.linalg.norm(g_est)
    if g_norm < 1e-8:
        raise ValueError("gravity estimate is degenerate")
    g_hat = g_est / g_norm
    residual = float(1.0 - g_ref @ g_hat)
    H = np.zeros(N_ERR)
    H[3:6] = -(g_ref @ (np.eye(3) - np.outer(g_hat, g_hat))) / (g_norm * dt)
    return residual, H, g_est


def _hat(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def stack_residuals(
    entries: list[tuple[float, np.ndarray, float]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_list, H_list, var_list = zip(*entries)
    return np.array(r_list), np.vstack(H_list), np.array(var_list)


def ekf_update(
    state: ImuState,
    P: np.ndarray,
    r: np.ndarray,
    H: np.ndarray,
    noise_var: np.ndarray,
) -> tuple[ImuState, np.ndarray]:
    R_mat = np.diag(noise_var)
    S = H @ P @ H.T + R_mat
    K = P @ H.T @ np.linalg.inv(S)
    dx = -K @ r
    new_state = state.boxplus(dx)

    I_KH = np.eye(N_ERR) - K @ H
    new_P = I_KH @ P @ I_KH.T + K @ R_mat @ K.T
    return new_state, new_P


def iterated_ekf_update(
    state0: ImuState,
    P0: np.ndarray,
    residual_fn: Callable[[ImuState], tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_iters: int = 4,
) -> tuple[ImuState, np.ndarray]:
    state, P = state0, P0
    for _ in range(n_iters):
        r, H, var = residual_fn(state)
        state, P = ekf_update(state, P, r, H, var)
    return state, P

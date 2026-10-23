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
) -> tuple[float, np.ndarray]:
    p_lidar = np.asarray(p_lidar, dtype=float).reshape(3)
    normal = np.asarray(normal, dtype=float).reshape(3)
    centroid = np.asarray(centroid, dtype=float).reshape(3)

    p_global = state.R @ p_lidar + state.p
    r = float(normal @ (p_global - centroid))

    H = np.zeros(N_ERR)
    H[0:3] = -(normal @ state.R @ hat(p_lidar))
    H[6:9] = normal
    return r, H


def radar_doppler_residual(
    state: ImuState,
    p_radar: np.ndarray,
    doppler_measured: float,
) -> tuple[float, np.ndarray]:
    p_radar = np.asarray(p_radar, dtype=float).reshape(3)
    u = p_radar / np.linalg.norm(p_radar)

    v_body = state.R.T @ state.v
    predicted = -float(u @ v_body)
    r = predicted - float(doppler_measured)

    H = np.zeros(N_ERR)
    H[0:3] = -(u @ hat(v_body))
    H[3:6] = -(u @ state.R.T)
    return r, H


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

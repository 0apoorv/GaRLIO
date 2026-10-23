from dataclasses import dataclass

import numpy as np

from .so3 import Exp, hat
from .state import ImuState

N_ERR = 15


@dataclass
class ImuNoiseParams:
    gyro_noise: float
    accel_noise: float
    gyro_bias_rw: float
    accel_bias_rw: float


class ESKF:
    def __init__(self, gravity: np.ndarray, noise: ImuNoiseParams):
        self.gravity = np.asarray(gravity, dtype=float).reshape(3)
        self.noise = noise

    def propagate(
        self,
        state: ImuState,
        P: np.ndarray,
        gyro_meas: np.ndarray,
        accel_meas: np.ndarray,
        dt: float,
    ) -> tuple[ImuState, np.ndarray]:
        gyro_meas = np.asarray(gyro_meas, dtype=float).reshape(3)
        accel_meas = np.asarray(accel_meas, dtype=float).reshape(3)

        omega = gyro_meas - state.bg
        accel = accel_meas - state.ba
        R = state.R

        a_world = R @ accel + self.gravity

        new_state = ImuState(
            R=R @ Exp(omega * dt),
            v=state.v + a_world * dt,
            p=state.p + state.v * dt + 0.5 * a_world * dt * dt,
            bg=state.bg.copy(),
            ba=state.ba.copy(),
        )

        F = np.eye(N_ERR)
        F[0:3, 0:3] = Exp(-omega * dt)
        F[0:3, 9:12] = -np.eye(3) * dt
        F[3:6, 0:3] = -R @ hat(accel) * dt
        F[3:6, 12:15] = -R * dt
        F[6:9, 3:6] = np.eye(3) * dt

        G = np.zeros((N_ERR, 12))
        G[0:3, 0:3] = -np.eye(3)
        G[3:6, 3:6] = -R
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)

        Qc = np.diag(
            [self.noise.gyro_noise ** 2] * 3
            + [self.noise.accel_noise ** 2] * 3
            + [self.noise.gyro_bias_rw ** 2] * 3
            + [self.noise.accel_bias_rw ** 2] * 3
        )
        Qd = G @ Qc @ G.T * dt

        new_P = F @ P @ F.T + Qd
        return new_state, new_P

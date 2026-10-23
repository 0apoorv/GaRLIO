import numpy as np
import pytest

from garlio.eskf import ESKF, ImuNoiseParams, N_ERR
from garlio.so3 import Exp
from garlio.state import ImuState

GRAVITY = np.array([0.0, 0.0, -9.81])

QUIET_NOISE = ImuNoiseParams(
    gyro_noise=1e-3,
    accel_noise=1e-2,
    gyro_bias_rw=1e-5,
    accel_bias_rw=1e-4,
)


def make_filter():
    return ESKF(gravity=GRAVITY, noise=QUIET_NOISE)


def test_stationary_imu_keeps_state_fixed():
    eskf = make_filter()
    state = ImuState()
    P = np.eye(N_ERR) * 1e-6

    accel_meas = np.array([0.0, 0.0, 9.81])
    gyro_meas = np.zeros(3)

    for _ in range(100):
        state, P = eskf.propagate(state, P, gyro_meas, accel_meas, dt=0.01)

    assert np.allclose(state.R, np.eye(3), atol=1e-9)
    assert np.allclose(state.v, np.zeros(3), atol=1e-9)
    assert np.allclose(state.p, np.zeros(3), atol=1e-9)


def test_constant_angular_velocity_matches_closed_form():
    eskf = make_filter()
    state = ImuState()
    P = np.eye(N_ERR) * 1e-6

    omega = np.array([0.0, 0.0, 1.0])
    accel_meas = np.array([0.0, 0.0, 9.81])
    dt = 0.001
    steps = 1000

    for _ in range(steps):
        state, P = eskf.propagate(state, P, omega, accel_meas, dt=dt)

    expected_R = Exp(omega * dt * steps)
    assert np.allclose(state.R, expected_R, atol=1e-4)


def test_free_fall_under_gravity_only():
    eskf = ESKF(gravity=GRAVITY, noise=QUIET_NOISE)
    state = ImuState()
    P = np.eye(N_ERR) * 1e-6

    accel_meas = np.zeros(3)
    gyro_meas = np.zeros(3)
    dt = 0.01
    steps = 100
    t_total = dt * steps

    for _ in range(steps):
        state, P = eskf.propagate(state, P, gyro_meas, accel_meas, dt=dt)

    expected_v = GRAVITY * t_total
    expected_p = 0.5 * GRAVITY * t_total ** 2
    assert np.allclose(state.v, expected_v, atol=1e-6)
    assert np.allclose(state.p, expected_p, atol=1e-6)


def test_covariance_stays_symmetric_and_grows():
    eskf = make_filter()
    state = ImuState()
    P = np.eye(N_ERR) * 1e-6

    accel_meas = np.array([0.3, -0.1, 9.9])
    gyro_meas = np.array([0.05, -0.02, 0.01])

    trace_history = [np.trace(P)]
    for _ in range(50):
        state, P = eskf.propagate(state, P, gyro_meas, accel_meas, dt=0.01)
        assert np.allclose(P, P.T, atol=1e-10)
        trace_history.append(np.trace(P))

    assert all(
        b >= a - 1e-12 for a, b in zip(trace_history, trace_history[1:])
    )


def test_covariance_is_positive_semidefinite():
    eskf = make_filter()
    state = ImuState()
    P = np.eye(N_ERR) * 1e-6

    accel_meas = np.array([0.2, 0.1, 9.7])
    gyro_meas = np.array([0.01, 0.03, -0.02])

    for _ in range(30):
        state, P = eskf.propagate(state, P, gyro_meas, accel_meas, dt=0.01)

    eigvals = np.linalg.eigvalsh(P)
    assert np.all(eigvals >= -1e-9)


def test_zero_dt_is_a_no_op_for_nominal_state():
    eskf = make_filter()
    state = ImuState(
        R=Exp(np.array([0.1, 0.2, 0.3])),
        v=np.array([1.0, 2.0, 3.0]),
        p=np.array([4.0, 5.0, 6.0]),
    )
    P = np.eye(N_ERR) * 1e-4

    accel_meas = np.array([0.1, 0.2, 9.9])
    gyro_meas = np.array([0.01, 0.02, 0.03])

    new_state, new_P = eskf.propagate(state, P, gyro_meas, accel_meas, dt=0.0)

    assert np.allclose(new_state.R, state.R)
    assert np.allclose(new_state.v, state.v)
    assert np.allclose(new_state.p, state.p)


def test_bias_offsets_gyro_and_accel_before_use():
    eskf = make_filter()
    bg_true = np.array([0.02, -0.01, 0.005])
    ba_true = np.array([0.1, -0.05, 0.2])
    state = ImuState(bg=bg_true.copy(), ba=ba_true.copy())
    P = np.eye(N_ERR) * 1e-6

    gyro_meas = bg_true.copy()
    accel_meas = np.array([0.0, 0.0, 9.81]) + ba_true

    state, P = eskf.propagate(state, P, gyro_meas, accel_meas, dt=0.01)

    assert np.allclose(state.R, np.eye(3), atol=1e-9)
    assert np.allclose(state.v, np.zeros(3), atol=1e-9)

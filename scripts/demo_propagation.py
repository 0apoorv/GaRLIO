import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from garlio.eskf import ESKF, ImuNoiseParams, N_ERR
from garlio.so3 import Exp
from garlio.state import ImuState

GRAVITY = np.array([0.0, 0.0, -9.81])
RADIUS = 5.0
OMEGA = 0.5
DT = 1.0 / 200.0
DURATION = 12.56


def ground_truth(t):
    yaw = OMEGA * t + np.pi / 2
    p = np.array([RADIUS * np.cos(OMEGA * t), RADIUS * np.sin(OMEGA * t), 0.0])
    v = RADIUS * OMEGA * np.array([-np.sin(OMEGA * t), np.cos(OMEGA * t), 0.0])
    a = RADIUS * OMEGA ** 2 * np.array([-np.cos(OMEGA * t), -np.sin(OMEGA * t), 0.0])
    R = Exp(np.array([0.0, 0.0, yaw]))
    return R, v, p, a


def make_measurements(t, bg_bias=np.zeros(3), ba_bias=np.zeros(3), rng=None):
    R, v, p, a_true = ground_truth(t)
    accel_meas = R.T @ (a_true - GRAVITY)
    gyro_meas = np.array([0.0, 0.0, OMEGA])
    accel_meas = accel_meas + ba_bias
    gyro_meas = gyro_meas + bg_bias
    if rng is not None:
        accel_meas = accel_meas + rng.normal(scale=0.05, size=3)
        gyro_meas = gyro_meas + rng.normal(scale=0.01, size=3)
    return gyro_meas, accel_meas


def run_propagation(bg_bias, ba_bias, add_noise):
    steps = int(DURATION / DT)
    R0, v0, p0, _ = ground_truth(0.0)
    state = ImuState(R=R0, v=v0, p=p0)
    P = np.eye(N_ERR) * 1e-8

    noise = ImuNoiseParams(
        gyro_noise=1e-3, accel_noise=1e-2, gyro_bias_rw=1e-5, accel_bias_rw=1e-4
    )
    eskf = ESKF(gravity=GRAVITY, noise=noise)
    rng = np.random.default_rng(0) if add_noise else None

    positions = [state.p.copy()]
    for i in range(steps):
        t = i * DT
        gyro_meas, accel_meas = make_measurements(t, bg_bias, ba_bias, rng)
        state, P = eskf.propagate(state, P, gyro_meas, accel_meas, DT)
        positions.append(state.p.copy())
    return np.array(positions)


def main():
    steps = int(DURATION / DT)
    t_arr = np.arange(steps + 1) * DT
    gt_positions = np.array([ground_truth(t)[2] for t in t_arr])

    clean = run_propagation(bg_bias=np.zeros(3), ba_bias=np.zeros(3), add_noise=False)
    clean_error = np.linalg.norm(clean[-1] - gt_positions[-1])
    print(f"Clean (noiseless) propagation final position error: {clean_error:.6f} m")
    assert clean_error < 0.1, "Noiseless propagation should track ground truth closely"

    noisy = run_propagation(
        bg_bias=np.array([0.01, -0.005, 0.008]),
        ba_bias=np.array([0.1, 0.05, -0.08]),
        add_noise=True,
    )
    noisy_error = np.linalg.norm(noisy[-1] - gt_positions[-1])
    print(f"Realistic (biased + noisy) IMU-only dead-reckoning final error: {noisy_error:.3f} m")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], "k-", linewidth=2, label="Ground truth")
    ax.plot(clean[:, 0], clean[:, 1], "g--", linewidth=1.5, label="ESKF propagation (noiseless IMU)")
    ax.plot(noisy[:, 0], noisy[:, 1], "r-", linewidth=1.2, label="IMU-only dead reckoning (realistic noise + bias)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("ESKF State Propagation: Circular Trajectory Demo")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    out_path = Path(__file__).resolve().parents[1] / "outputs" / "propagation_demo.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()

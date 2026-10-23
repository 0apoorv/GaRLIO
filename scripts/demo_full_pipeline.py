import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from garlio.eskf import ESKF, ImuNoiseParams, N_ERR
from garlio.so3 import Exp
from garlio.state import ImuState
from garlio.update import point_to_plane_residual, radar_doppler_residual, stack_residuals, iterated_ekf_update

GRAVITY = np.array([0.0, 0.0, -9.81])
RADIUS = 5.0
OMEGA = 0.5
DT = 1.0 / 200.0
DURATION = 12.56
UPDATE_EVERY_N_STEPS = 20

ROOM_HALF_SIZE = 8.0
WALL_HEIGHT = 3.0


def ground_truth(t):
    yaw = OMEGA * t + np.pi / 2
    p = np.array([RADIUS * np.cos(OMEGA * t), RADIUS * np.sin(OMEGA * t), 0.0])
    v = RADIUS * OMEGA * np.array([-np.sin(OMEGA * t), np.cos(OMEGA * t), 0.0])
    a = RADIUS * OMEGA ** 2 * np.array([-np.cos(OMEGA * t), -np.sin(OMEGA * t), 0.0])
    R = Exp(np.array([0.0, 0.0, yaw]))
    return R, v, p, a


def make_imu_measurements(t, bg_bias, ba_bias, rng):
    R, v, p, a_true = ground_truth(t)
    accel_meas = R.T @ (a_true - GRAVITY) + ba_bias + rng.normal(scale=0.05, size=3)
    gyro_meas = np.array([0.0, 0.0, OMEGA]) + bg_bias + rng.normal(scale=0.01, size=3)
    return gyro_meas, accel_meas


def make_room_lidar_points(true_R, true_p, rng, n_per_wall=15):
    walls = [
        (np.array([1.0, 0.0, 0.0]), np.array([ROOM_HALF_SIZE, 0.0, WALL_HEIGHT / 2])),
        (np.array([-1.0, 0.0, 0.0]), np.array([-ROOM_HALF_SIZE, 0.0, WALL_HEIGHT / 2])),
        (np.array([0.0, 1.0, 0.0]), np.array([0.0, ROOM_HALF_SIZE, WALL_HEIGHT / 2])),
        (np.array([0.0, -1.0, 0.0]), np.array([0.0, -ROOM_HALF_SIZE, WALL_HEIGHT / 2])),
        (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0])),
    ]
    entries = []
    for normal, centroid in walls:
        if abs(normal[0]) > 0.5:
            y = rng.uniform(-ROOM_HALF_SIZE, ROOM_HALF_SIZE, n_per_wall)
            z = rng.uniform(0, WALL_HEIGHT, n_per_wall)
            world_pts = np.column_stack([np.full(n_per_wall, centroid[0]), y, z])
        elif abs(normal[1]) > 0.5:
            x = rng.uniform(-ROOM_HALF_SIZE, ROOM_HALF_SIZE, n_per_wall)
            z = rng.uniform(0, WALL_HEIGHT, n_per_wall)
            world_pts = np.column_stack([x, np.full(n_per_wall, centroid[1]), z])
        else:
            x = rng.uniform(-ROOM_HALF_SIZE, ROOM_HALF_SIZE, n_per_wall)
            y = rng.uniform(-ROOM_HALF_SIZE, ROOM_HALF_SIZE, n_per_wall)
            world_pts = np.column_stack([x, y, np.zeros(n_per_wall)])

        body_pts = (true_R.T @ (world_pts - true_p).T).T
        entries.append((body_pts, normal, centroid))
    return entries


def make_radar_points(true_R, true_v, rng, n=25):
    radar_pts = rng.uniform(-1, 1, size=(n, 3))
    radar_pts[:, 2] = rng.uniform(-0.3, 0.3, n)
    u = radar_pts / np.linalg.norm(radar_pts, axis=1, keepdims=True)
    v_body_true = true_R.T @ true_v
    doppler = -u @ v_body_true + rng.normal(scale=0.02, size=n)
    return radar_pts, doppler


def run(use_corrections):
    steps = int(DURATION / DT)
    R0, v0, p0, _ = ground_truth(0.0)
    state = ImuState(R=R0, v=v0, p=p0)
    P = np.eye(N_ERR) * 1e-6

    noise = ImuNoiseParams(
        gyro_noise=1e-3, accel_noise=1e-2, gyro_bias_rw=1e-5, accel_bias_rw=1e-4
    )
    eskf = ESKF(gravity=GRAVITY, noise=noise)
    rng = np.random.default_rng(7)

    bg_bias = np.array([0.01, -0.005, 0.008])
    ba_bias = np.array([0.1, 0.05, -0.08])

    positions = [state.p.copy()]
    for i in range(steps):
        t = i * DT
        gyro_meas, accel_meas = make_imu_measurements(t, bg_bias, ba_bias, rng)
        state, P = eskf.propagate(state, P, gyro_meas, accel_meas, DT)

        if use_corrections and (i + 1) % UPDATE_EVERY_N_STEPS == 0:
            true_R, true_v, true_p, _ = ground_truth(t + DT)
            room = make_room_lidar_points(true_R, true_p, rng)
            radar_pts, doppler = make_radar_points(true_R, true_v, rng)

            def residual_fn(s):
                lidar_entries = [
                    point_to_plane_residual(s, p, normal, centroid) + (0.02,)
                    for body_pts, normal, centroid in room
                    for p in body_pts
                ]
                radar_entries = [
                    radar_doppler_residual(s, radar_pts[j], doppler[j]) + (0.03,)
                    for j in range(len(doppler))
                ]
                return stack_residuals(lidar_entries + radar_entries)

            state, P = iterated_ekf_update(state, P, residual_fn, n_iters=4)

        positions.append(state.p.copy())
    return np.array(positions)


def main():
    steps = int(DURATION / DT)
    t_arr = np.arange(steps + 1) * DT
    gt_positions = np.array([ground_truth(t)[2] for t in t_arr])

    dead_reckoning = run(use_corrections=False)
    corrected = run(use_corrections=True)

    dr_error = np.linalg.norm(dead_reckoning[-1] - gt_positions[-1])
    corrected_error = np.linalg.norm(corrected[-1] - gt_positions[-1])
    print(f"IMU-only dead reckoning final error: {dr_error:.3f} m")
    print(f"GaRLIO-style corrected final error:  {corrected_error:.3f} m")
    assert corrected_error < dr_error * 0.1, "corrections should drastically reduce drift"

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], "k-", linewidth=2, label="Ground truth")
    ax.plot(dead_reckoning[:, 0], dead_reckoning[:, 1], "r-", linewidth=1.2, label="IMU-only dead reckoning")
    ax.plot(corrected[:, 0], corrected[:, 1], "b--", linewidth=1.5, label="With LiDAR+radar corrections")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Full Pipeline: Propagation + Update Corrections")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    out_path = Path(__file__).resolve().parents[1] / "outputs" / "full_pipeline_demo.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()

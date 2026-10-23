import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from garlio.eskf import ESKF, ImuNoiseParams, N_ERR
from garlio.state import ImuState
from garlio.mapping import LocalPlaneMap
from garlio.radar import estimate_ego_velocity_ransac
from garlio.update import point_to_plane_residual, radar_doppler_residual, stack_residuals, iterated_ekf_update

GRAVITY = np.array([0.0, 0.0, -9.81])
MAP_SUBSAMPLE = 5000
QUERY_SUBSAMPLE = 300
K_NEIGHBORS = 12
PLANARITY_THRESH = 0.05


def load_data(data_dir: Path):
    imu = np.load(data_dir / "imu.npy")
    radar = np.load(data_dir / "radar.npz", allow_pickle=True)
    lidar = np.load(data_dir / "lidar.npz", allow_pickle=True)
    return imu, radar, lidar


def propagate_imu_segment(eskf, state, P, imu, t_start, t_end):
    mask = (imu[:, 0] > t_start) & (imu[:, 0] <= t_end)
    segment = imu[mask]
    prev_t = t_start
    for row in segment:
        t = row[0]
        dt = t - prev_t
        if dt <= 0:
            prev_t = t
            continue
        accel_meas = row[1:4]
        gyro_meas = row[4:7]
        state, P = eskf.propagate(state, P, gyro_meas, accel_meas, dt)
        prev_t = t
    return state, P


def transform_to_world(points, state):
    return (state.R @ points.T).T + state.p


def find_nearest_radar_frame(radar_times, radar_frames, t_query):
    idx = np.argmin(np.abs(radar_times - t_query))
    return radar_frames[idx]


def run_pipeline(imu, radar, lidar, use_corrections: bool, rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    noise = ImuNoiseParams(gyro_noise=1e-3, accel_noise=1e-2, gyro_bias_rw=1e-5, accel_bias_rw=1e-4)
    eskf = ESKF(gravity=GRAVITY, noise=noise)

    lidar_times = lidar["times"]
    lidar_frames = lidar["frames"]
    radar_times = radar["times"]
    radar_frames = radar["frames"]

    state = ImuState()
    P = np.eye(N_ERR) * 1e-6

    t0 = lidar_times[0]
    state, P = propagate_imu_segment(eskf, state, P, imu, imu[0, 0], t0)

    positions = [state.p.copy()]
    prev_map = None
    prev_t = t0

    for i in range(1, len(lidar_times)):
        t = lidar_times[i]
        state, P = propagate_imu_segment(eskf, state, P, imu, prev_t, t)

        current_frame = lidar_frames[i]
        if len(current_frame) > MAP_SUBSAMPLE:
            map_idx = rng.choice(len(current_frame), MAP_SUBSAMPLE, replace=False)
        else:
            map_idx = np.arange(len(current_frame))
        world_map_points = transform_to_world(current_frame[map_idx], state)

        if use_corrections and prev_map is not None:
            n_query = min(QUERY_SUBSAMPLE, len(current_frame))
            query_idx = rng.choice(len(current_frame), n_query, replace=False)
            query_points = current_frame[query_idx]

            radar_frame = find_nearest_radar_frame(radar_times, radar_frames, t)
            radar_result = None
            if len(radar_frame) >= 5:
                radar_pts = radar_frame[:, :3]
                doppler = radar_frame[:, 3]
                try:
                    radar_result = estimate_ego_velocity_ransac(
                        radar_pts, doppler, inlier_thresh=0.3, n_iters=100, rng=rng
                    )
                except ValueError:
                    radar_result = None

            def residual_fn(s):
                entries = []
                for p in query_points:
                    p_world = s.R @ p + s.p
                    fit = prev_map.query(p_world)
                    if fit is not None:
                        r, H = point_to_plane_residual(s, p, fit.normal, fit.centroid)
                        entries.append((r, H, 0.05))
                if radar_result is not None:
                    static_pts = radar_pts[radar_result.static_mask]
                    static_doppler = doppler[radar_result.static_mask]
                    n_use = min(50, len(static_pts))
                    if n_use > 0:
                        use_idx = rng.choice(len(static_pts), n_use, replace=False)
                        for j in use_idx:
                            r, H = radar_doppler_residual(s, static_pts[j], static_doppler[j])
                            entries.append((r, H, 0.1))
                if not entries:
                    return np.zeros(0), np.zeros((0, N_ERR)), np.zeros(0)
                return stack_residuals(entries)

            r0, H0, var0 = residual_fn(state)
            if len(r0) >= 6:
                state, P = iterated_ekf_update(state, P, residual_fn, n_iters=3)

        prev_map = LocalPlaneMap(world_map_points, k_neighbors=K_NEIGHBORS, planarity_thresh=PLANARITY_THRESH)
        prev_t = t
        positions.append(state.p.copy())

    return np.array(positions), lidar_times


def main():
    data_dir = Path(__file__).resolve().parents[1] / "real_data" / "extracted_data"
    imu, radar, lidar = load_data(data_dir)
    print(f"Loaded {len(imu)} IMU samples, {len(radar['frames'])} radar frames, {len(lidar['frames'])} lidar frames")

    dead_reckoning, times_dr = run_pipeline(imu, radar, lidar, use_corrections=False)
    corrected, times_c = run_pipeline(imu, radar, lidar, use_corrections=True)

    print(f"Dead-reckoning final position: {dead_reckoning[-1]}")
    print(f"Corrected final position:      {corrected[-1]}")
    print(f"Dead-reckoning path length (approx): {np.sum(np.linalg.norm(np.diff(dead_reckoning, axis=0), axis=1)):.2f} m")
    print(f"Corrected path length (approx):      {np.sum(np.linalg.norm(np.diff(corrected, axis=0), axis=1)):.2f} m")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    ax.plot(dead_reckoning[:, 0], dead_reckoning[:, 1], "r-", linewidth=1.2, label="IMU-only dead reckoning")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("IMU-only (no corrections)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(corrected[:, 0], corrected[:, 1], "b-", linewidth=1.2, label="With LiDAR+radar corrections")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("With LiDAR+radar corrections")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Real Snail-Radar Sequence: IMU-only vs Corrected Trajectory")
    fig.tight_layout()

    out_path = Path(__file__).resolve().parents[1] / "outputs" / "real_data_pipeline_demo.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()

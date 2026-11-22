import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from garlio.eskf import ESKF, ImuNoiseParams, N_ERR
from garlio.state import ImuState
from garlio.mapping import LocalPlaneMap
from garlio.radar import estimate_ego_velocity_ransac, modified_z_scores
from garlio.update import (
    point_to_plane_residual, radar_doppler_residual, gravity_direction_residual,
    stack_residuals, iterated_ekf_update,
)
from garlio.snail_radar_extrinsics import T_X36D_XT32, T_X36D_OCULII

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
    return propagate_imu_samples(eskf, state, P, segment, t_start)


def propagate_imu_samples(eskf, state, P, samples, t_start):
    prev_t = t_start
    R_start = state.R.copy()
    delta_v_body = np.zeros(3)
    for row in samples:
        t = row[0]
        dt = t - prev_t
        if dt <= 0:
            prev_t = t
            continue
        accel_meas = row[1:4]
        gyro_meas = row[4:7]
        delta_v_body += R_start.T @ state.R @ (accel_meas - state.ba) * dt
        state, P = eskf.propagate(state, P, gyro_meas, accel_meas, dt)
        prev_t = t
    return state, P, delta_v_body


def transform_to_world(points, state, T_imu_sensor):
    points_imu = (T_imu_sensor[:3, :3] @ points.T).T + T_imu_sensor[:3, 3]
    return (state.R @ points_imu.T).T + state.p


def find_nearest_radar_frame(radar_times, radar_frames, t_query):
    idx = np.argmin(np.abs(radar_times - t_query))
    return radar_frames[idx]


def angular_velocity_at(imu, t_query):
    """Return the most recent raw gyro sample at or before ``t_query``."""
    index = int(np.searchsorted(imu[:, 0], t_query, side="right") - 1)
    return imu[max(index, 0), 4:7]


class PipelineProcessor:
    """Incremental form of the existing scan update for batch and stream input."""

    def __init__(
        self, use_corrections: bool, rng_seed: int = 0,
        use_gravity_update: bool = True, use_radar_update: bool = True,
        imu_start_time: float | None = None,
    ):
        self.rng = np.random.default_rng(rng_seed)
        noise = ImuNoiseParams(gyro_noise=1e-3, accel_noise=1e-2, gyro_bias_rw=1e-5, accel_bias_rw=1e-4)
        self.eskf = ESKF(gravity=GRAVITY, noise=noise)
        self.state = ImuState()
        self.P = np.eye(N_ERR) * 1e-6
        self.use_corrections = use_corrections
        self.use_gravity_update = use_gravity_update
        self.use_radar_update = use_radar_update
        self.imu_start_time = imu_start_time
        self.prev_t = None
        self.prev_map = None

    def process_scan(
        self, t: float, current_frame: np.ndarray, imu_samples: np.ndarray,
        radar_frame: np.ndarray | None, gyro_at_scan: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        first_scan = self.prev_t is None
        if first_scan:
            start = self.imu_start_time
            if start is None:
                start = float(imu_samples[0, 0]) if len(imu_samples) else t
            self.state, self.P, _ = propagate_imu_samples(
                self.eskf, self.state, self.P, imu_samples, start
            )
        else:
            previous_state = self.state.copy()
            self.state, self.P, delta_v_body = propagate_imu_samples(
                self.eskf, self.state, self.P, imu_samples, self.prev_t
            )

        # The original batch runner initializes the first pose but does not
        # use the first scan to build a map or consume any point-sampling RNG.
        if first_scan:
            self.prev_t = t
            return self.state.p.copy(), self.state.R.copy()

        if len(current_frame) > MAP_SUBSAMPLE:
            map_idx = self.rng.choice(len(current_frame), MAP_SUBSAMPLE, replace=False)
        else:
            map_idx = np.arange(len(current_frame))
        world_map_points = transform_to_world(current_frame[map_idx], self.state, T_X36D_XT32)

        if self.use_corrections and self.prev_map is not None:
            n_query = min(QUERY_SUBSAMPLE, len(current_frame))
            query_idx = self.rng.choice(len(current_frame), n_query, replace=False)
            query_points = current_frame[query_idx]
            radar_result = None
            radar_pts = np.zeros((0, 3))
            doppler = np.zeros(0)
            if radar_frame is not None and len(radar_frame) >= 5:
                radar_pts = radar_frame[:, :3]
                doppler = radar_frame[:, 3]
                try:
                    radar_result = estimate_ego_velocity_ransac(
                        radar_pts, doppler, inlier_thresh=0.3, n_iters=100, rng=self.rng
                    )
                except ValueError:
                    radar_result = None

            def residual_fn(s):
                entries = []
                for p in query_points:
                    p_imu = T_X36D_XT32[:3, :3] @ p + T_X36D_XT32[:3, 3]
                    p_world = s.R @ p_imu + s.p
                    fit = self.prev_map.query(p_world)
                    if fit is not None:
                        r, H = point_to_plane_residual(s, p, fit.normal, fit.centroid, T_X36D_XT32)
                        entries.append((r, H, 0.05))
                if radar_result is not None and self.use_radar_update and gyro_at_scan is not None:
                    static_pts = radar_pts[radar_result.static_mask]
                    static_doppler = doppler[radar_result.static_mask]
                    static_unit = static_pts / np.maximum(
                        np.linalg.norm(static_pts, axis=1, keepdims=True), 1e-9
                    )
                    static_residuals = np.abs(-static_unit @ radar_result.v_ego - static_doppler)
                    relative_R = self.state.R @ previous_state.R.T
                    turn_angle = np.arccos(np.clip((np.trace(relative_R) - 1.0) / 2.0, -1.0, 1.0))
                    quality_residuals = static_residuals * np.exp(min(10.0 * turn_angle, 50.0))
                    static_scores = modified_z_scores(quality_residuals)
                    ranked_idx = np.argsort(quality_residuals)
                    n_use = min(50, max(1, len(static_pts) // 3))
                    for j in ranked_idx[:n_use]:
                        r, H = radar_doppler_residual(
                            s, static_pts[j], static_doppler[j], T_X36D_OCULII, gyro_at_scan
                        )
                        excess = max(0.0, abs(static_scores[j]) - 3.5)
                        variance = 0.1 * (1.0 + excess / 5.0)
                        entries.append((r, H, variance))
                if not entries:
                    return np.zeros(0), np.zeros((0, N_ERR)), np.zeros(0)
                return stack_residuals(entries)

            r0, _, _ = residual_fn(self.state)
            if len(r0) >= 6:
                self.state, self.P = iterated_ekf_update(
                    self.state, self.P, residual_fn, n_iters=3
                )

            if self.use_gravity_update:
                try:
                    dt = t - self.prev_t

                    def gravity_fn(s):
                        r_g, H_g, _ = gravity_direction_residual(
                            s, previous_state, delta_v_body, GRAVITY, dt
                        )
                        return np.array([r_g]), H_g.reshape(1, -1), np.array([0.01])

                    gravity_est = gravity_direction_residual(
                        self.state, previous_state, delta_v_body, GRAVITY, dt
                    )[2]
                    gravity_angle = np.arccos(np.clip(
                        (GRAVITY / np.linalg.norm(GRAVITY)) @
                        (gravity_est / np.linalg.norm(gravity_est)), -1.0, 1.0
                    ))
                    if gravity_angle < 0.15:
                        self.state, self.P = iterated_ekf_update(
                            self.state, self.P, gravity_fn, n_iters=1
                        )
                except ValueError:
                    pass

        self.prev_map = LocalPlaneMap(
            world_map_points, k_neighbors=K_NEIGHBORS,
            planarity_thresh=PLANARITY_THRESH,
        )
        self.prev_t = t
        return self.state.p.copy(), self.state.R.copy()


def run_pipeline(
    imu, radar, lidar, use_corrections: bool, rng_seed: int = 0,
    return_rotations: bool = False, use_gravity_update: bool = True,
    use_radar_update: bool = True,
):
    lidar_times = lidar["times"]
    lidar_frames = lidar["frames"]
    radar_times = radar["times"]
    radar_frames = radar["frames"]
    processor = PipelineProcessor(
        use_corrections, rng_seed, use_gravity_update, use_radar_update,
        imu_start_time=float(imu[0, 0]),
    )
    positions, rotations = [], []
    imu_index = 0
    for t, frame in zip(lidar_times, lidar_frames):
        next_imu_index = int(np.searchsorted(imu[imu_index:, 0], t, side="right")) + imu_index
        imu_samples = imu[imu_index:next_imu_index]
        imu_index = next_imu_index
        radar_frame = radar_frames[int(np.argmin(np.abs(radar_times - t)))] if len(radar_times) else None
        gyro = angular_velocity_at(imu, t) if len(imu) else None
        position, rotation = processor.process_scan(t, frame, imu_samples, radar_frame, gyro)
        positions.append(position)
        rotations.append(rotation)

    positions = np.asarray(positions)
    rotations = np.asarray(rotations)
    if return_rotations:
        return positions, rotations, lidar_times
    return positions, lidar_times


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

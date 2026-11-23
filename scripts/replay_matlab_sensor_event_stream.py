"""Consume MATLAB's sorted event manifest with the incremental Python estimator."""
import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_real_data_pipeline import PipelineProcessor
from garlio.snail_radar_extrinsics import imu_poses_to_xt32

MAX_BUFFERED_IMU_SAMPLES = 10000
MAX_PENDING_LIDAR_SCANS = 100


def _cell_frames(value):
    return [np.asarray(frame, dtype=np.float32).T for frame in np.asarray(value, dtype=object).ravel(order="F")]


def replay_matlab_stream(source_path: Path, event_path: Path, output_dir: Path,
                         max_radar_dt: float = 0.06, rng_seed: int = 0,
                         reference_mat: Path | None = None):
    source = loadmat(source_path, squeeze_me=False)
    events = loadmat(event_path, squeeze_me=False)
    imu_times = np.asarray(source["imu_times"], dtype=float).reshape(-1)
    imu_values = np.asarray(source["imu_values"], dtype=float)
    radar_times = np.asarray(source["radar_times"], dtype=float).reshape(-1)
    radar_frames = _cell_frames(source["radar_frames"])
    lidar_times = np.asarray(source["lidar_times"], dtype=float).reshape(-1)
    lidar_frames = _cell_frames(source["lidar_frames"])
    event_times = np.asarray(events["eventTimes"], dtype=float).reshape(-1)
    event_types = np.asarray(events["eventTypes"], dtype=np.uint8).reshape(-1)
    event_indices = np.asarray(events["eventIndices"], dtype=np.int64).reshape(-1) - 1

    if not (len(event_times) == len(event_types) == len(event_indices)):
        raise ValueError("MATLAB event arrays have different lengths.")
    if np.any(np.diff(event_times) < 0):
        raise ValueError("MATLAB event stream is not sorted by timestamp.")
    if imu_values.shape != (6, len(imu_times)):
        raise ValueError(f"Expected imu_values shape (6, N), got {imu_values.shape}.")
    if len(radar_frames) != len(radar_times) or len(lidar_frames) != len(lidar_times):
        raise ValueError("Frame cell counts do not match their timestamp counts.")

    output_dir.mkdir(parents=True, exist_ok=True)
    processor = None
    imu_buffer = []
    latest_gyro = None
    pending_lidar = []
    previous_radar = None
    latest_radar = None
    first_time = float(event_times[0])
    simulated_finish = 0.0
    max_imu_buffer = 0
    max_lidar_pending = 0
    skipped_radar = 0
    scan_rows = []
    rotations = []

    def process_ready(radar_after=None, final=False):
        nonlocal processor, imu_buffer, latest_gyro, previous_radar, simulated_finish
        nonlocal skipped_radar
        while pending_lidar:
            t_lidar, lidar_frame = pending_lidar[0]
            bracket_ready = radar_after is not None and radar_after[0] >= t_lidar
            if not bracket_ready and not final:
                break
            candidates = [r for r in (previous_radar, radar_after or latest_radar) if r is not None]
            radar_frame = None
            radar_dt = np.inf
            if candidates:
                nearest = min(candidates, key=lambda r: abs(r[0] - t_lidar))
                radar_dt = abs(nearest[0] - t_lidar)
                if radar_dt <= max_radar_dt:
                    radar_frame = nearest[1]
                else:
                    skipped_radar += 1

            imu_samples_list = [row for row in imu_buffer if row[0] <= t_lidar]
            imu_buffer = [row for row in imu_buffer if row[0] > t_lidar]
            if imu_samples_list:
                latest_gyro = imu_samples_list[-1][4:7]
            if processor is None:
                imu_start = imu_samples_list[0][0] if imu_samples_list else t_lidar
                processor = PipelineProcessor(
                    True, rng_seed, True, True, imu_start_time=float(imu_start)
                )
            imu_samples = np.asarray(imu_samples_list, dtype=np.float64).reshape(-1, 7)

            started = time.perf_counter()
            position, rotation = processor.process_scan(
                t_lidar, lidar_frame, imu_samples, radar_frame, latest_gyro
            )
            service = time.perf_counter() - started
            rotations.append(rotation)

            ready_time = radar_after[0] if bracket_ready else t_lidar
            ready_elapsed = max(0.0, ready_time - first_time)
            queue_wait = max(0.0, simulated_finish - ready_elapsed)
            finish = max(simulated_finish, ready_elapsed) + service
            latency = finish - max(0.0, t_lidar - first_time)
            scan_rows.append({
                "timestamp_s": t_lidar,
                "position_x_m": position[0],
                "position_y_m": position[1],
                "position_z_m": position[2],
                "service_time_s": service,
                "radar_time_offset_s": radar_dt,
                "radar_used": radar_frame is not None,
                "sensor_wait_s": max(0.0, ready_time - t_lidar),
                "queue_wait_s": queue_wait,
                "simulated_output_latency_s": latency,
                "simulated_finish_elapsed_s": finish,
            })
            simulated_finish = finish
            pending_lidar.pop(0)
        if radar_after is not None:
            previous_radar = radar_after

    for t, kind, idx in zip(event_times, event_types, event_indices):
        if kind == 1:
            row = np.concatenate(([t], imu_values[:, idx]))
            imu_buffer.append(row)
            max_imu_buffer = max(max_imu_buffer, len(imu_buffer))
            if len(imu_buffer) > MAX_BUFFERED_IMU_SAMPLES:
                raise RuntimeError("IMU buffer limit exceeded; check MATLAB event ordering.")
        elif kind == 2:
            latest_radar = (float(t), radar_frames[idx])
            process_ready(radar_after=latest_radar)
        elif kind == 3:
            pending_lidar.append((float(t), lidar_frames[idx]))
            max_lidar_pending = max(max_lidar_pending, len(pending_lidar))
            if len(pending_lidar) > MAX_PENDING_LIDAR_SCANS:
                raise RuntimeError("LiDAR queue limit exceeded; check radar events/order.")
            if latest_radar is not None and latest_radar[0] >= t:
                process_ready(radar_after=latest_radar)
            else:
                process_ready()
        else:
            raise ValueError(f"Unknown MATLAB event type code {kind}.")

    process_ready(final=True)
    if not scan_rows:
        raise RuntimeError("MATLAB event stream contained no LiDAR scans.")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "matlab_stream_timing.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scan_rows[0]))
        writer.writeheader()
        writer.writerows(scan_rows)

    scan_times = np.array([r["timestamp_s"] for r in scan_rows])
    service = np.array([r["service_time_s"] for r in scan_rows])
    queue = np.array([r["queue_wait_s"] for r in scan_rows])
    latency = np.array([r["simulated_output_latency_s"] for r in scan_rows])
    periods = np.diff(scan_times)
    relative_times = scan_times - scan_times[0]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(relative_times, service, linewidth=0.8, label="Measured processing")
    axes[0].axhline(np.median(periods), color="tab:red", linestyle="--",
                    label=f"Median LiDAR period ({np.median(periods):.3f} s)")
    axes[0].set_ylabel("Seconds per scan")
    axes[0].set_title("MATLAB-generated sensor stream: Python estimator timing")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(relative_times, queue, color="tab:orange", linewidth=1.0)
    axes[1].set_xlabel("Recorded sequence time (s)")
    axes[1].set_ylabel("Simulated queue wait (s)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    plot_path = output_dir / "matlab_stream_timing.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    positions = np.array([[r["position_x_m"], r["position_y_m"], r["position_z_m"]]
                          for r in scan_rows])
    np.savez_compressed(output_dir / "matlab_stream_trajectory.npz",
                        times=scan_times, positions=positions, rotations=np.asarray(rotations))
    finish = np.array([r["simulated_finish_elapsed_s"] for r in scan_rows])
    misses = int(np.sum(finish[:-1] > scan_times[1:] - first_time))
    print(f"Processed {len(scan_rows)} LiDAR scans from {len(event_times)} MATLAB events.")
    print(f"Scans without radar within {max_radar_dt:.3f}s: {skipped_radar}")
    print(f"Processing mean={service.mean():.4f}s, p50={np.percentile(service, 50):.4f}s, "
          f"p95={np.percentile(service, 95):.4f}s, max={service.max():.4f}s")
    print(f"Schedule misses={misses}/{max(0, len(scan_rows)-1)}; "
          f"output latency p50={np.percentile(latency, 50):.4f}s, "
          f"p95={np.percentile(latency, 95):.4f}s; max queue wait={queue.max():.4f}s")
    print(f"Maximum buffers: IMU={max_imu_buffer} samples, LiDAR={max_lidar_pending} scans")
    print(f"Saved {csv_path}, matlab_stream_trajectory.npz, and {plot_path}")
    if reference_mat is not None and reference_mat.exists():
        reference = loadmat(reference_mat, squeeze_me=False)
        reference_times = np.asarray(reference["times_corrected"], dtype=float).reshape(-1)
        reference_positions = np.asarray(reference["positions_corrected"], dtype=float).T
        reference_rotations = np.moveaxis(
            np.asarray(reference["rotations_corrected"], dtype=float), 2, 0
        )
        converted_positions, converted_rotations = imu_poses_to_xt32(
            positions, np.asarray(rotations)
        )
        n = len(scan_times)
        times_match = np.allclose(scan_times, reference_times[:n], atol=1e-8)
        position_max = float(np.max(np.abs(converted_positions - reference_positions[:n])))
        rotation_max = float(np.max(np.abs(converted_rotations - reference_rotations[:n])))
        print(
            f"Batch parity ({n} scans): timestamps_match={times_match}, "
            f"max position difference={position_max:.3e} m, "
            f"max rotation difference={rotation_max:.3e}"
        )
    return scan_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_mat", type=Path)
    parser.add_argument("event_mat", type=Path,
                        help="MATLAB output from generate_sensor_event_stream.m")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/matlab_stream_run"))
    parser.add_argument("--max-radar-dt", type=float, default=0.06)
    parser.add_argument("--reference-mat", type=Path,
                        default=Path("outputs/full_sequence/trajectory_evaluation.mat"))
    args = parser.parse_args()
    replay_matlab_stream(args.source_mat, args.event_mat, args.output_dir,
                         args.max_radar_dt, reference_mat=args.reference_mat)


if __name__ == "__main__":
    main()

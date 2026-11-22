"""Timestamp-ordered, offline replay of a ROS bag through the Python estimator.

This is a real-time scheduling simulation: recorded messages are consumed in
timestamp order, but this script does not subscribe to live ROS topics.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rosbags.highlevel import AnyReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_real_data_pipeline import PipelineProcessor
from extract_snail_radar import (
    IMU_TOPICS, LIDAR_TOPIC, RADAR_TOPICS, decode_pointcloud2,
)

MAX_BUFFERED_IMU_SAMPLES = 10000
MAX_PENDING_LIDAR_SCANS = 100


def choose_connections(reader):
    imu = [c for c in reader.connections if c.topic in IMU_TOPICS]
    lidar = [c for c in reader.connections if c.topic == LIDAR_TOPIC]
    radar = []
    doppler_name = None
    for topic, field_name in RADAR_TOPICS:
        radar = [c for c in reader.connections if c.topic == topic]
        if radar:
            doppler_name = field_name
            break
    if not imu or not lidar or not radar:
        found = sorted({c.topic for c in reader.connections})
        raise RuntimeError(
            "Required IMU, radar, or LiDAR topic was not found. "
            f"Selected counts: IMU={len(imu)}, radar={len(radar)}, LiDAR={len(lidar)}. "
            f"Bag topics: {found}"
        )
    return imu, radar, lidar, doppler_name


def decode_imu(msg, timestamp_s):
    return np.array([
        timestamp_s,
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
    ], dtype=np.float64)


def decode_cloud(msg, timestamp_s, radar=False, doppler_name=None):
    fields = decode_pointcloud2(msg)
    if radar:
        if doppler_name not in fields:
            raise RuntimeError(
                f"Radar cloud has no '{doppler_name}' Doppler field; fields={sorted(fields)}"
            )
        points = np.column_stack([
            fields["x"], fields["y"], fields["z"], fields[doppler_name]
        ])
    else:
        points = np.column_stack([fields["x"], fields["y"], fields["z"]])
    return timestamp_s, points.astype(np.float32, copy=False)


def replay_bag(
    bag_path: Path,
    output_dir: Path,
    max_seconds: float | None = None,
    max_radar_dt: float = 0.06,
    rng_seed: int = 0,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    with AnyReader([bag_path]) as reader:
        imu_conns, radar_conns, lidar_conns, doppler_name = choose_connections(reader)
        conns = imu_conns + radar_conns + lidar_conns
        print(
            "Topics: "
            f"IMU={imu_conns[0].topic}, radar={radar_conns[0].topic}, "
            f"LiDAR={lidar_conns[0].topic}"
        )

        processor = None
        imu_buffer = []
        latest_gyro = None
        pending_lidar = []
        previous_radar = None
        latest_radar = None
        first_sensor_time = None
        previous_message_time = None
        simulated_finish_elapsed = 0.0
        scan_rows = []
        rotation_matrices = []
        max_imu_buffered = 0
        max_lidar_pending = 0
        counts = {"imu": 0, "radar": 0, "lidar": 0, "skipped_radar": 0}

        def process_ready(radar_after=None, final=False):
            nonlocal processor, imu_buffer, latest_gyro, simulated_finish_elapsed
            nonlocal previous_radar
            while pending_lidar:
                lidar_t, lidar_frame = pending_lidar[0]
                bracket_ready = (
                    radar_after is not None and radar_after[0] >= lidar_t
                )
                if not bracket_ready and not final:
                    break
                candidates = [r for r in (previous_radar, radar_after or latest_radar) if r is not None]
                radar_frame = None
                nearest_dt = np.inf
                if candidates:
                    chosen = min(candidates, key=lambda r: abs(r[0] - lidar_t))
                    nearest_dt = abs(chosen[0] - lidar_t)
                    if nearest_dt <= max_radar_dt:
                        radar_frame = chosen[1]
                    else:
                        counts["skipped_radar"] += 1

                available_imu = [row for row in imu_buffer if row[0] <= lidar_t]
                imu_buffer = [row for row in imu_buffer if row[0] > lidar_t]
                if available_imu:
                    latest_gyro = available_imu[-1][4:7]
                if processor is None:
                    imu_start = float(available_imu[0][0]) if available_imu else lidar_t
                    processor = PipelineProcessor(
                        use_corrections=True,
                        rng_seed=rng_seed,
                        use_gravity_update=True,
                        use_radar_update=True,
                        imu_start_time=imu_start,
                    )
                samples = np.asarray(available_imu, dtype=np.float64).reshape(-1, 7)
                gyro = latest_gyro

                service_started = time.perf_counter()
                position, rotation = processor.process_scan(
                    lidar_t, lidar_frame, samples, radar_frame, gyro
                )
                service_s = time.perf_counter() - service_started
                rotation_matrices.append(rotation)

                # The scan can only be evaluated once the next radar timestamp
                # brackets it. The scheduler then models a single serial worker.
                ready_t = radar_after[0] if bracket_ready else lidar_t
                ready_elapsed = max(0.0, ready_t - first_sensor_time)
                queue_wait_s = max(0.0, simulated_finish_elapsed - ready_elapsed)
                start_elapsed = max(simulated_finish_elapsed, ready_elapsed)
                finish_elapsed = start_elapsed + service_s
                output_latency_s = finish_elapsed - max(0.0, lidar_t - first_sensor_time)
                scan_rows.append({
                    "timestamp_s": lidar_t,
                    "position_x_m": position[0],
                    "position_y_m": position[1],
                    "position_z_m": position[2],
                    "service_time_s": service_s,
                    "radar_time_offset_s": nearest_dt,
                    "radar_used": radar_frame is not None,
                    "sensor_wait_s": max(0.0, ready_t - lidar_t),
                    "queue_wait_s": queue_wait_s,
                    "simulated_output_latency_s": output_latency_s,
                    "simulated_finish_elapsed_s": finish_elapsed,
                })
                simulated_finish_elapsed = finish_elapsed
                pending_lidar.pop(0)
            if radar_after is not None:
                previous_radar = radar_after

        for connection, timestamp_ns, rawdata in reader.messages(connections=conns):
            t = timestamp_ns * 1e-9
            if previous_message_time is not None and t < previous_message_time:
                raise RuntimeError(
                    f"Bag messages are not timestamp ordered: {t} follows {previous_message_time}"
                )
            previous_message_time = t
            if first_sensor_time is None:
                first_sensor_time = t
            if max_seconds is not None and t - first_sensor_time > max_seconds:
                break
            msg = reader.deserialize(rawdata, connection.msgtype)
            if connection in imu_conns:
                imu_buffer.append(decode_imu(msg, t))
                counts["imu"] += 1
                max_imu_buffered = max(max_imu_buffered, len(imu_buffer))
                if len(imu_buffer) > MAX_BUFFERED_IMU_SAMPLES:
                    raise RuntimeError(
                        f"IMU buffer exceeded its {MAX_BUFFERED_IMU_SAMPLES}-sample limit; "
                        "check radar availability and timestamp alignment."
                    )
            elif connection in radar_conns:
                radar = decode_cloud(msg, t, radar=True, doppler_name=doppler_name)
                latest_radar = radar
                counts["radar"] += 1
                process_ready(radar_after=radar)
            elif connection in lidar_conns:
                pending_lidar.append(decode_cloud(msg, t, radar=False))
                counts["lidar"] += 1
                max_lidar_pending = max(max_lidar_pending, len(pending_lidar))
                if len(pending_lidar) > MAX_PENDING_LIDAR_SCANS:
                    raise RuntimeError(
                        f"Pending LiDAR queue exceeded its {MAX_PENDING_LIDAR_SCANS}-scan limit; "
                        "check radar availability and timestamp alignment."
                    )
                process_ready()

        process_ready(final=True)

    if not scan_rows:
        raise RuntimeError("No LiDAR scans were processed from this bag.")

    csv_path = output_dir / "stream_replay_timing.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scan_rows[0]))
        writer.writeheader()
        writer.writerows(scan_rows)
    relative_times = np.array([r["timestamp_s"] for r in scan_rows])
    relative_times -= relative_times[0]
    service_times = np.array([r["service_time_s"] for r in scan_rows])
    scan_periods = np.diff([r["timestamp_s"] for r in scan_rows])
    queue_waits = np.array([r["queue_wait_s"] for r in scan_rows])
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(relative_times, service_times, color="tab:blue", linewidth=0.8,
                 label="Measured processing time")
    axes[0].axhline(np.median(scan_periods), color="tab:red", linestyle="--",
                    label=f"Median LiDAR period ({np.median(scan_periods):.3f} s)")
    axes[0].set_ylabel("Seconds per scan")
    axes[0].set_title("Python bag replay: processing time and simulated backlog")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(relative_times, queue_waits, color="tab:orange", linewidth=1.0)
    axes[1].set_xlabel("Recorded sequence time (s)")
    axes[1].set_ylabel("Simulated queue wait (s)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    plot_path = output_dir / "stream_replay_timing.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(
        output_dir / "stream_replay_trajectory.npz",
        times=np.array([r["timestamp_s"] for r in scan_rows]),
        positions=np.array([[r["position_x_m"], r["position_y_m"], r["position_z_m"]] for r in scan_rows]),
        rotations=np.asarray(rotation_matrices),
    )

    service = np.array([r["service_time_s"] for r in scan_rows])
    latency = np.array([r["simulated_output_latency_s"] for r in scan_rows])
    queue_wait = np.array([r["queue_wait_s"] for r in scan_rows])
    scan_times = np.array([r["timestamp_s"] for r in scan_rows])
    interscan = np.diff(scan_times)
    deadline_misses = np.sum(
        np.array([r["simulated_finish_elapsed_s"] for r in scan_rows[:-1]])
        > (scan_times[1:] - first_sensor_time)
    )
    print(f"Processed {len(scan_rows)} LiDAR scans from {counts['lidar']} recorded scans.")
    print(
        f"Sensor messages: IMU={counts['imu']}, radar={counts['radar']}; "
        f"scans without radar within {max_radar_dt:.3f}s={counts['skipped_radar']}"
    )
    print(
        f"LiDAR period median={np.median(interscan):.4f}s; measured processing "
        f"mean={np.mean(service):.4f}s, p50={np.percentile(service, 50):.4f}s, "
        f"p95={np.percentile(service, 95):.4f}s, "
        f"max={np.max(service):.4f}s"
    )
    print(
        f"Timestamp-schedule simulation: output latency p50={np.percentile(latency, 50):.4f}s, "
        f"p95={np.percentile(latency, 95):.4f}s, max={np.max(latency):.4f}s; "
        f"queue wait max={np.max(queue_wait):.4f}s; "
        f"deadline misses={deadline_misses}/{max(0, len(scan_rows) - 1)} inter-scan deadlines"
    )
    print(
        f"Maximum input buffers: {max_imu_buffered} IMU samples, "
        f"{max_lidar_pending} pending LiDAR scans"
    )
    print(f"Saved per-scan timing: {csv_path}")
    print(f"Saved timing plot: {plot_path}")
    return scan_rows


def main():
    parser = argparse.ArgumentParser(
        description="Replay a ROS bag in timestamp order through the Python GaRLIO pipeline"
    )
    parser.add_argument("bag_path", type=Path, help="Path to the recorded .bag file")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/stream_replay"),
        help="Where to save per-scan timing CSV and pose trajectory",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=None,
        help="Optional duration from the first selected sensor message",
    )
    parser.add_argument(
        "--max-radar-dt", type=float, default=0.06,
        help="Maximum time offset allowed when matching a radar frame to a LiDAR scan",
    )
    args = parser.parse_args()
    replay_bag(args.bag_path, args.output_dir, args.max_seconds, args.max_radar_dt)


if __name__ == "__main__":
    main()

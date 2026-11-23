"""Wall-clock-paced replay of a MATLAB event manifest through the GaRLIO pipeline.

The MATLAB file provides the timestamp-ordered IMU/radar/LiDAR schedule. Raw
measurements remain in the companion source MAT file. A producer thread emits
events at their recorded rate while the estimator consumes them independently,
so queueing and deadline misses are measured from wall-clock execution.

This is a recorded-data real-time simulation, not a live sensor or ROS test.
"""
import argparse
import csv
import queue
import sys
import threading
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
END_OF_STREAM = object()


def _cell_frames(value):
    return [np.asarray(frame, dtype=np.float32).T
            for frame in np.asarray(value, dtype=object).ravel(order="F")]


def replay_realtime(source_path: Path, event_path: Path, output_dir: Path,
                    playback_speed: float = 1.0, max_radar_dt: float = 0.06,
                    rng_seed: int = 0,
                    reference_mat: Path | None = None) -> list[dict]:
    if playback_speed <= 0:
        raise ValueError("playback_speed must be greater than zero")

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
    if len(event_times) == 0 or np.any(np.diff(event_times) < 0):
        raise ValueError("MATLAB event stream is empty or not timestamp ordered.")
    if imu_values.shape != (6, len(imu_times)):
        raise ValueError(f"Expected imu_values shape (6, N), got {imu_values.shape}.")
    if len(radar_frames) != len(radar_times) or len(lidar_frames) != len(lidar_times):
        raise ValueError("Frame cell counts do not match their timestamp counts.")

    # Fail early on bad type/index pairs rather than silently indexing the
    # wrong payload from the companion raw sensor MAT file.
    bounds = {1: len(imu_times), 2: len(radar_times), 3: len(lidar_times)}
    for kind in (1, 2, 3):
        mask = event_types == kind
        if not np.any(mask):
            continue
        indices = event_indices[mask]
        if np.any(indices < 0) or np.any(indices >= bounds[kind]):
            raise ValueError(f"Event type {kind} contains an out-of-range source index.")
        source_t = (imu_times if kind == 1 else radar_times if kind == 2 else lidar_times)
        if not np.allclose(event_times[mask], source_t[indices], atol=1e-8, rtol=0):
            raise ValueError(f"Event timestamps do not match source type {kind} indices.")

    output_dir.mkdir(parents=True, exist_ok=True)
    playback_origin = time.perf_counter()
    first_time = float(event_times[0])
    event_queue: queue.Queue = queue.Queue(maxsize=20000)
    producer_error: list[BaseException] = []
    producer_stats = {"max_queue": 0, "scheduled_events": 0, "late_events": 0}

    def producer():
        try:
            for t, kind, idx in zip(event_times, event_types, event_indices):
                due = playback_origin + (float(t) - first_time) / playback_speed
                remaining = due - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                emitted = time.perf_counter()
                event_queue.put((float(t), int(kind), int(idx), due, emitted))
                producer_stats["scheduled_events"] += 1
                producer_stats["late_events"] += int(emitted - due > 0.01)
                producer_stats["max_queue"] = max(
                    producer_stats["max_queue"], event_queue.qsize()
                )
        except BaseException as exc:  # propagate thread failures to the caller
            producer_error.append(exc)
        finally:
            event_queue.put(END_OF_STREAM)

    thread = threading.Thread(target=producer, name="matlab-stream-producer", daemon=True)
    thread.start()

    processor = None
    imu_buffer = []
    latest_gyro = None
    pending_lidar = []
    previous_radar = None
    latest_radar = None
    scan_rows: list[dict] = []
    rotations = []
    counts = {"imu": 0, "radar": 0, "lidar": 0, "radar_skipped": 0}
    max_imu_buffer = 0
    max_lidar_pending = 0
    consumer_max_queue = 0

    def process_ready(radar_after=None, final=False):
        nonlocal processor, imu_buffer, latest_gyro, previous_radar
        while pending_lidar:
            t_lidar, lidar_frame, lidar_due, lidar_emitted = pending_lidar[0]
            bracket_ready = radar_after is not None and radar_after[0] >= t_lidar
            if not bracket_ready and not final:
                break

            candidates = [r for r in (previous_radar, radar_after or latest_radar)
                          if r is not None]
            radar_frame = None
            radar_dt = float("inf")
            if candidates:
                nearest = min(candidates, key=lambda r: abs(r[0] - t_lidar))
                radar_dt = abs(nearest[0] - t_lidar)
                if radar_dt <= max_radar_dt:
                    radar_frame = nearest[1]
                else:
                    counts["radar_skipped"] += 1

            imu_rows = [row for row in imu_buffer if row[0] <= t_lidar]
            imu_buffer = [row for row in imu_buffer if row[0] > t_lidar]
            if imu_rows:
                latest_gyro = imu_rows[-1][4:7]
            if processor is None:
                imu_start = imu_rows[0][0] if imu_rows else t_lidar
                processor = PipelineProcessor(
                    True, rng_seed, True, True, imu_start_time=float(imu_start)
                )

            samples = np.asarray(imu_rows, dtype=np.float64).reshape(-1, 7)
            start_wall = time.perf_counter()
            position, rotation = processor.process_scan(
                t_lidar, lidar_frame, samples, radar_frame, latest_gyro
            )
            finish_wall = time.perf_counter()
            rotations.append(rotation)

            next_scan_idx = len(scan_rows) + 1
            next_scan_time = (float(lidar_times[next_scan_idx])
                              if next_scan_idx < len(lidar_times) else np.nan)
            next_due = (playback_origin + (next_scan_time - first_time) / playback_speed
                        if np.isfinite(next_scan_time) else np.nan)
            scan_rows.append({
                "timestamp_s": t_lidar,
                "position_x_m": position[0],
                "position_y_m": position[1],
                "position_z_m": position[2],
                "sensor_wait_s": max(0.0, start_wall - lidar_due),
                "queue_wait_s": max(0.0, start_wall - lidar_emitted),
                "service_time_s": finish_wall - start_wall,
                "output_latency_s": finish_wall - lidar_due,
                "radar_time_offset_s": radar_dt,
                "radar_used": radar_frame is not None,
                "next_scan_deadline_s": next_due,
                "deadline_missed": bool(np.isfinite(next_due) and finish_wall > next_due),
            })
            pending_lidar.pop(0)

        if radar_after is not None:
            previous_radar = radar_after

    while True:
        event = event_queue.get()
        if event is END_OF_STREAM:
            break
        t, kind, idx, due, emitted = event
        consumer_max_queue = max(consumer_max_queue, event_queue.qsize())
        if kind == 1:
            imu_buffer.append(np.concatenate(([t], imu_values[:, idx])))
            counts["imu"] += 1
            max_imu_buffer = max(max_imu_buffer, len(imu_buffer))
            if max_imu_buffer > MAX_BUFFERED_IMU_SAMPLES:
                raise RuntimeError("IMU buffer exceeded its safety limit.")
        elif kind == 2:
            radar = (t, radar_frames[idx], emitted)
            latest_radar = radar
            counts["radar"] += 1
            process_ready(radar_after=radar)
        elif kind == 3:
            pending_lidar.append((t, lidar_frames[idx], due, emitted))
            counts["lidar"] += 1
            max_lidar_pending = max(max_lidar_pending, len(pending_lidar))
            if max_lidar_pending > MAX_PENDING_LIDAR_SCANS:
                raise RuntimeError("Pending LiDAR queue exceeded its safety limit.")
            process_ready()
        else:
            raise ValueError(f"Unknown MATLAB event type code {kind}.")

    process_ready(final=True)
    thread.join()
    if producer_error:
        raise RuntimeError("The timestamped event producer failed.") from producer_error[0]
    if not scan_rows:
        raise RuntimeError("MATLAB event stream contained no LiDAR scans.")

    csv_path = output_dir / "matlab_realtime_timing.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scan_rows[0]))
        writer.writeheader()
        writer.writerows(scan_rows)

    times = np.array([r["timestamp_s"] for r in scan_rows])
    service = np.array([r["service_time_s"] for r in scan_rows])
    queue_wait = np.array([r["queue_wait_s"] for r in scan_rows])
    latency = np.array([r["output_latency_s"] for r in scan_rows])
    periods = np.diff(times) / playback_speed
    relative_times = (times - times[0]) / playback_speed
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(relative_times, service, linewidth=0.8, label="Measured estimator service time")
    axes[0].axhline(np.median(periods), color="tab:red", linestyle="--",
                    label=f"Median LiDAR period ({np.median(periods):.3f} s)")
    axes[0].set_ylabel("Seconds per scan")
    axes[0].set_title("Wall-clock-paced MATLAB sensor schedule through GaRLIO")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(relative_times, queue_wait, color="tab:orange", label="Input queue wait")
    axes[1].plot(relative_times, latency, color="tab:green", alpha=0.8,
                 label="Output latency from scheduled LiDAR time")
    axes[1].set_xlabel("Recorded sequence time (s)")
    axes[1].set_ylabel("Seconds")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    plot_path = output_dir / "matlab_realtime_timing.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    positions = np.array([[r["position_x_m"], r["position_y_m"], r["position_z_m"]]
                          for r in scan_rows])
    np.savez_compressed(output_dir / "matlab_realtime_trajectory.npz",
                        times=times, positions=positions, rotations=np.asarray(rotations))

    misses = int(sum(r["deadline_missed"] for r in scan_rows))
    elapsed = time.perf_counter() - playback_origin
    print(f"Wall-clock-paced playback: {elapsed:.3f}s for "
          f"{(event_times[-1]-event_times[0])/playback_speed:.3f}s of sensor time.")
    print(f"Estimator consumed {counts['imu']} IMU, {counts['radar']} radar, "
          f"and {counts['lidar']} LiDAR events; processed {len(scan_rows)} scans.")
    print(f"Radar omissions (> {max_radar_dt:.3f}s): {counts['radar_skipped']}.")
    print(f"Scan service mean={service.mean():.4f}s, p50={np.percentile(service,50):.4f}s, "
          f"p95={np.percentile(service,95):.4f}s, max={service.max():.4f}s.")
    print(f"Missed next-scan deadlines={misses}/{len(scan_rows)-1}; "
          f"output latency p50={np.percentile(latency,50):.4f}s, "
          f"p95={np.percentile(latency,95):.4f}s, max={latency.max():.4f}s.")
    print(f"Max observed event queue: producer={producer_stats['max_queue']}, "
          f"consumer={consumer_max_queue}; producer events >10ms late="
          f"{producer_stats['late_events']}/{producer_stats['scheduled_events']}.")
    print(f"Maximum estimator buffers: IMU={max_imu_buffer} samples, "
          f"pending LiDAR={max_lidar_pending} scans; playback speed={playback_speed:g}x.")
    print(f"Saved {csv_path}, matlab_realtime_trajectory.npz, and {plot_path}.")

    if reference_mat is not None and reference_mat.exists():
        reference = loadmat(reference_mat, squeeze_me=False)
        ref_times = np.asarray(reference["times_corrected"], dtype=float).reshape(-1)
        ref_positions = np.asarray(reference["positions_corrected"], dtype=float).T
        ref_rotations = np.moveaxis(
            np.asarray(reference["rotations_corrected"], dtype=float), 2, 0
        )
        converted_positions, converted_rotations = imu_poses_to_xt32(
            positions, np.asarray(rotations)
        )
        n = len(times)
        print("Batch parity ({} scans): timestamps_match={}, "
              "max position difference={:.3e}m, max rotation difference={:.3e}.".format(
                  n, np.allclose(times, ref_times[:n], atol=1e-8),
                  np.max(np.abs(converted_positions - ref_positions[:n])),
                  np.max(np.abs(converted_rotations - ref_rotations[:n]))))
    return scan_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_mat", type=Path)
    parser.add_argument("event_mat", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/matlab_sensor_stream/realtime_run"))
    parser.add_argument("--playback-speed", type=float, default=1.0,
                        help="1.0 is recorded-time pacing; values above 1 run faster.")
    parser.add_argument("--max-radar-dt", type=float, default=0.06)
    parser.add_argument("--reference-mat", type=Path,
                        default=Path("outputs/full_sequence/trajectory_evaluation.mat"))
    args = parser.parse_args()
    replay_realtime(args.source_mat, args.event_mat, args.output_dir,
                    args.playback_speed, args.max_radar_dt,
                    reference_mat=args.reference_mat)


if __name__ == "__main__":
    main()

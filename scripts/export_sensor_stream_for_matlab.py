"""Export extracted Snail-Radar sensor frames in a MATLAB-friendly MAT file."""
import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat


def export(data_dir: Path, output_path: Path, max_seconds: float | None = None):
    imu = np.load(data_dir / "imu.npy")
    radar_file = np.load(data_dir / "radar.npz", allow_pickle=True)
    lidar_file = np.load(data_dir / "lidar.npz", allow_pickle=True)

    lidar_t0 = float(lidar_file["times"][0])
    if max_seconds is None:
        lidar_end = float(lidar_file["times"][-1])
    else:
        lidar_end = lidar_t0 + max_seconds
    # Keep sensor samples from sequence start through the selected LiDAR end.
    # Pre-scan IMU and radar measurements are needed to initialize/match the
    # first scan with the same history available to the batch evaluator.
    radar_mask = radar_file["times"] <= lidar_end
    lidar_mask = (lidar_file["times"] >= lidar_t0) & (lidar_file["times"] <= lidar_end)
    imu_mask = imu[:, 0] <= lidar_end

    radar_frames = np.empty((1, int(radar_mask.sum())), dtype=object)
    for i, frame in enumerate(radar_file["frames"][radar_mask]):
        radar_frames[0, i] = np.asarray(frame, dtype=np.float32).T
    lidar_frames = np.empty((1, int(lidar_mask.sum())), dtype=object)
    for i, frame in enumerate(lidar_file["frames"][lidar_mask]):
        lidar_frames[0, i] = np.asarray(frame, dtype=np.float32).T

    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        output_path,
        {
            "imu_times": imu[imu_mask, 0].reshape(1, -1),
            "imu_values": imu[imu_mask, 1:7].T,
            "radar_times": radar_file["times"][radar_mask].reshape(1, -1),
            "radar_frames": radar_frames,
            "lidar_times": lidar_file["times"][lidar_mask].reshape(1, -1),
            "lidar_frames": lidar_frames,
            "sensor_time_origin": min(float(imu[0, 0]), float(radar_file["times"][0]), lidar_t0),
            "max_seconds_exported": -1.0 if max_seconds is None else max_seconds,
        },
        do_compression=True,
        long_field_names=True,
    )
    lidar_points = sum(len(frame) for frame in lidar_file["frames"][lidar_mask])
    print(
        f"Wrote {output_path}: IMU={int(imu_mask.sum())}, "
        f"radar={int(radar_mask.sum())}, LiDAR={int(lidar_mask.sum())} frames, "
        f"LiDAR points={lidar_points}"
    )


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("/Users/apoorvsharma/Downloads/garlio_btp_full/real_data/extracted_data"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=repo_root / "outputs" / "matlab_sensor_stream" / "sensor_stream_source.mat",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=15.0,
        help="Export the first N seconds (default 15); pass 0 for the full sequence.",
    )
    args = parser.parse_args()
    export(args.data_dir, args.output, None if args.max_seconds == 0 else args.max_seconds)


if __name__ == "__main__":
    main()

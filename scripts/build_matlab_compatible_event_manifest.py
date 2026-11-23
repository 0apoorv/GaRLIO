"""Build the same compact event manifest as matlab/generate_sensor_event_stream.m.

This local helper is useful for large sequences whose point-cloud MAT file
would be inconvenient to upload to MATLAB Drive. The manifest contains event
times, type codes (1=IMU, 2=radar, 3=LiDAR), and one-based source indices. Its
ordering is equivalent to MATLAB sortrows([time,type,index],[1,2,3]).
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


def build_manifest(source_path: Path, output_path: Path) -> None:
    source = loadmat(source_path, variable_names=[
        "imu_times", "radar_times", "lidar_times"
    ])
    parts = [
        (np.asarray(source[name], dtype=float).reshape(-1), kind)
        for name, kind in (("imu_times", 1), ("radar_times", 2), ("lidar_times", 3))
    ]
    if any(times.size == 0 for times, _ in parts):
        raise ValueError("Each sensor timestamp array must contain at least one event.")

    event_times = np.concatenate([times for times, _ in parts])
    event_types = np.concatenate([
        np.full(times.size, kind, dtype=np.uint8) for times, kind in parts
    ])
    event_indices = np.concatenate([
        np.arange(1, times.size + 1, dtype=np.uint32) for times, _ in parts
    ])

    # np.lexsort uses its last key as primary, matching MATLAB's ascending
    # sortrows key order [time, type, index].
    order = np.lexsort((event_indices, event_types, event_times))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        output_path,
        {
            "eventTimes": event_times[order],
            "eventTypes": event_types[order],
            "eventIndices": event_indices[order],
        },
        do_compression=True,
        format="5",
    )
    print(
        f"Wrote MATLAB-compatible manifest: {len(order)} events; "
        f"IMU={(event_types[order] == 1).sum()}, "
        f"radar={(event_types[order] == 2).sum()}, "
        f"LiDAR={(event_types[order] == 3).sum()}; "
        f"span={event_times[order][-1] - event_times[order][0]:.3f}s; "
        f"file={output_path} ({output_path.stat().st_size:,} bytes)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_mat", type=Path)
    parser.add_argument("output_mat", type=Path)
    args = parser.parse_args()
    build_manifest(args.source_mat, args.output_mat)


if __name__ == "__main__":
    main()

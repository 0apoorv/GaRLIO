import argparse
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

ROS_DATATYPE_TO_NUMPY = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}

IMU_TOPICS = ["/x36d/imu_raw", "/mti3dk/imu", "/zed2i/zed_node/imu/data"]
RADAR_TOPICS = [
    ("/ars548", "doppler"),
    ("/radar_pcl2", "Doppler"),
    ("/radar_enhanced_pcl2", "Doppler"),
]
LIDAR_TOPIC = "/hesai/pandar"


def decode_pointcloud2(msg) -> dict[str, np.ndarray]:
    point_step = msg.point_step
    n_points = msg.width * msg.height
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n_points, point_step)

    out = {}
    for field in msg.fields:
        dtype = ROS_DATATYPE_TO_NUMPY[field.datatype]
        itemsize = np.dtype(dtype).itemsize
        col_bytes = raw[:, field.offset : field.offset + itemsize].copy()
        out[field.name] = col_bytes.view(dtype).reshape(n_points)
    return out


def extract(bag_path: str, out_dir: str, max_seconds: float | None = None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imu_rows = []
    radar_times = []
    radar_points = []
    lidar_times = []
    lidar_points = []

    with AnyReader([Path(bag_path)]) as reader:
        t0 = None

        conns_imu = [c for c in reader.connections if c.topic in IMU_TOPICS]

        conns_radar = []
        doppler_field = "doppler"
        for topic, field_name in RADAR_TOPICS:
            found = [c for c in reader.connections if c.topic == topic]
            if found:
                conns_radar = found
                doppler_field = field_name
                break

        conns_lidar = [c for c in reader.connections if c.topic == LIDAR_TOPIC]

        print(f"IMU topic:   {conns_imu[0].topic if conns_imu else 'NOT FOUND'}")
        print(f"Radar topic: {conns_radar[0].topic if conns_radar else 'NOT FOUND'}")
        print(f"Lidar topic: {LIDAR_TOPIC if conns_lidar else 'NOT FOUND'}")

        if conns_imu:
            for conn, timestamp, rawdata in reader.messages(connections=conns_imu):
                msg = reader.deserialize(rawdata, conn.msgtype)
                t = timestamp * 1e-9
                if t0 is None:
                    t0 = t
                if max_seconds is not None and (t - t0) > max_seconds:
                    continue
                imu_rows.append(
                    [
                        t,
                        msg.linear_acceleration.x,
                        msg.linear_acceleration.y,
                        msg.linear_acceleration.z,
                        msg.angular_velocity.x,
                        msg.angular_velocity.y,
                        msg.angular_velocity.z,
                    ]
                )

        if conns_radar:
            for conn, timestamp, rawdata in reader.messages(connections=conns_radar):
                msg = reader.deserialize(rawdata, conn.msgtype)
                t = timestamp * 1e-9
                if t0 is not None and max_seconds is not None and (t - t0) > max_seconds:
                    continue
                fields = decode_pointcloud2(msg)
                stacked = np.column_stack(
                    [fields["x"], fields["y"], fields["z"], fields[doppler_field]]
                )
                radar_times.append(t)
                radar_points.append(stacked.astype(np.float32))

        if conns_lidar:
            for conn, timestamp, rawdata in reader.messages(connections=conns_lidar):
                msg = reader.deserialize(rawdata, conn.msgtype)
                t = timestamp * 1e-9
                if t0 is not None and max_seconds is not None and (t - t0) > max_seconds:
                    continue
                fields = decode_pointcloud2(msg)
                stacked = np.column_stack([fields["x"], fields["y"], fields["z"]])
                lidar_times.append(t)
                lidar_points.append(stacked.astype(np.float32))

    imu_arr = np.array(imu_rows, dtype=np.float64)
    np.save(out_dir / "imu.npy", imu_arr)

    np.savez(
        out_dir / "radar.npz",
        times=np.array(radar_times, dtype=np.float64),
        frames=np.array(radar_points, dtype=object),
        allow_pickle=True,
    )
    np.savez(
        out_dir / "lidar.npz",
        times=np.array(lidar_times, dtype=np.float64),
        frames=np.array(lidar_points, dtype=object),
        allow_pickle=True,
    )

    print(
        f"Saved {len(imu_rows)} IMU samples, {len(radar_times)} radar frames, "
        f"{len(lidar_times)} lidar frames to {out_dir}"
    )
    return imu_arr, radar_times, radar_points, lidar_times, lidar_points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract IMU/radar/LiDAR topics from a Snail-Radar rosbag into compact .npy/.npz files"
    )
    parser.add_argument("bag_path", help="Path to the .bag file (unzipped from data1.bag.zip)")
    parser.add_argument("out_dir", help="Output directory for extracted files")
    parser.add_argument(
        "--max_seconds",
        type=float,
        default=15.0,
        help="Only extract the first N seconds (default 15, keeps file sizes small)",
    )
    args = parser.parse_args()
    extract(args.bag_path, args.out_dir, args.max_seconds)

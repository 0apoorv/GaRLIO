import numpy as np


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


def remove_dynamic_lidar_points(
    lidar_points: np.ndarray,
    dynamic_radar_points: np.ndarray,
    T_radar_from_lidar: np.ndarray,
    sigma_xy: float = 0.5,
    mahalanobis_thresh: float = 3.0,
) -> np.ndarray:
    lidar_points = np.asarray(lidar_points, dtype=float)
    n = lidar_points.shape[0]

    if dynamic_radar_points.shape[0] == 0:
        return np.ones(n, dtype=bool)

    lidar_in_radar = transform_points(lidar_points, T_radar_from_lidar)
    lidar_xy = lidar_in_radar[:, :2]
    radar_xy = np.asarray(dynamic_radar_points, dtype=float)[:, :2]

    cov_inv = np.eye(2) / (sigma_xy ** 2)

    diffs = lidar_xy[:, None, :] - radar_xy[None, :, :]
    mahal_sq = np.einsum("nmi,ij,nmj->nm", diffs, cov_inv, diffs)
    min_mahal_sq = mahal_sq.min(axis=1)

    is_dynamic = min_mahal_sq < mahalanobis_thresh ** 2
    return ~is_dynamic

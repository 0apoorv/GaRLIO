from dataclasses import dataclass

import numpy as np


@dataclass
class RadarEgoVelocityResult:
    v_ego: np.ndarray
    static_mask: np.ndarray
    dynamic_mask: np.ndarray


def _unit_vectors(points: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    return points / norms


def estimate_ego_velocity_ransac(
    points: np.ndarray,
    doppler: np.ndarray,
    inlier_thresh: float = 0.1,
    n_iters: int = 200,
    min_points: int = 3,
    rng: np.random.Generator | None = None,
) -> RadarEgoVelocityResult:
    points = np.asarray(points, dtype=float)
    doppler = np.asarray(doppler, dtype=float)
    n = points.shape[0]

    if n < min_points:
        raise ValueError(f"need at least {min_points} radar points, got {n}")

    if rng is None:
        rng = np.random.default_rng()

    u = _unit_vectors(points)
    A_full = -u

    best_inlier_mask = None
    best_count = -1

    for _ in range(n_iters):
        sample_idx = rng.choice(n, size=3, replace=False)
        A_sample = A_full[sample_idx]
        b_sample = doppler[sample_idx]

        if abs(np.linalg.det(A_sample)) < 1e-9:
            continue

        v_candidate = np.linalg.solve(A_sample, b_sample)
        residuals = np.abs(A_full @ v_candidate - doppler)
        inlier_mask = residuals < inlier_thresh
        count = int(inlier_mask.sum())

        if count > best_count:
            best_count = count
            best_inlier_mask = inlier_mask

    if best_inlier_mask is None or best_count < min_points:
        v_ego, *_ = np.linalg.lstsq(A_full, doppler, rcond=None)
        residuals = np.abs(A_full @ v_ego - doppler)
        static_mask = residuals < inlier_thresh
        return RadarEgoVelocityResult(
            v_ego=v_ego, static_mask=static_mask, dynamic_mask=~static_mask
        )

    A_inliers = A_full[best_inlier_mask]
    b_inliers = doppler[best_inlier_mask]
    v_ego, *_ = np.linalg.lstsq(A_inliers, b_inliers, rcond=None)

    residuals = np.abs(A_full @ v_ego - doppler)
    static_mask = residuals < inlier_thresh
    dynamic_mask = ~static_mask

    return RadarEgoVelocityResult(
        v_ego=v_ego, static_mask=static_mask, dynamic_mask=dynamic_mask
    )

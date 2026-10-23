from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class PlaneFit:
    normal: np.ndarray
    centroid: np.ndarray
    is_planar: bool
    planarity_ratio: float


def fit_local_plane(
    neighbor_points: np.ndarray,
    planarity_thresh: float = 0.1,
) -> PlaneFit:
    neighbor_points = np.asarray(neighbor_points, dtype=float)
    centroid = neighbor_points.mean(axis=0)
    centered = neighbor_points - centroid

    cov = centered.T @ centered / len(neighbor_points)
    eigvals, eigvecs = np.linalg.eigh(cov)

    normal = eigvecs[:, 0]
    if eigvals[1] < 1e-12:
        ratio = np.inf
    else:
        ratio = eigvals[0] / eigvals[1]
    is_planar = ratio < planarity_thresh

    return PlaneFit(normal=normal, centroid=centroid, is_planar=is_planar, planarity_ratio=ratio)


class LocalPlaneMap:
    def __init__(self, map_points: np.ndarray, k_neighbors: int = 5, planarity_thresh: float = 0.1):
        self.map_points = np.asarray(map_points, dtype=float)
        self.tree = cKDTree(self.map_points)
        self.k_neighbors = k_neighbors
        self.planarity_thresh = planarity_thresh

    def query(self, point_world: np.ndarray) -> PlaneFit | None:
        dists, idx = self.tree.query(point_world, k=self.k_neighbors)
        neighbors = self.map_points[idx]
        fit = fit_local_plane(neighbors, self.planarity_thresh)
        if not fit.is_planar:
            return None
        return fit

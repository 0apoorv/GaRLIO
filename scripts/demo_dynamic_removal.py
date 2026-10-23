import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from garlio.radar import estimate_ego_velocity_ransac
from garlio.dynamic_removal import remove_dynamic_lidar_points

rng = np.random.default_rng(42)

V_EGO_TRUE = np.array([3.0, 0.0, 0.0])

N_STATIC_BG = 150
static_bg = rng.uniform(-15, 15, size=(N_STATIC_BG, 3))
static_bg[:, 2] = rng.uniform(-0.5, 0.5, size=N_STATIC_BG)

car_center_start = np.array([6.0, 3.0, 0.0])
car_velocity = np.array([-4.0, 1.0, 0.0])
n_car_points = 25
car_points = car_center_start + rng.normal(scale=0.4, size=(n_car_points, 3))
car_points[:, 2] = rng.uniform(-0.3, 0.3, size=n_car_points)

lidar_points = np.vstack([static_bg, car_points])
n_static = static_bg.shape[0]

u = static_bg / np.linalg.norm(static_bg, axis=1, keepdims=True)
static_doppler = -u @ V_EGO_TRUE + rng.normal(scale=0.02, size=n_static)

u_car = car_points / np.linalg.norm(car_points, axis=1, keepdims=True)
relative_velocity = car_velocity - V_EGO_TRUE
car_doppler = -u_car @ V_EGO_TRUE - u_car @ (car_velocity)

radar_points = np.vstack([static_bg, car_points])
radar_doppler = np.concatenate([static_doppler, car_doppler])

result = estimate_ego_velocity_ransac(
    radar_points, radar_doppler, inlier_thresh=0.15, n_iters=300, rng=rng
)

print(f"True ego velocity:      {V_EGO_TRUE}")
print(f"Estimated ego velocity: {result.v_ego}")
print(f"Detected {result.dynamic_mask.sum()} dynamic radar points out of {len(radar_doppler)}")

dynamic_radar_points = radar_points[result.dynamic_mask]

keep_mask = remove_dynamic_lidar_points(
    lidar_points,
    dynamic_radar_points,
    T_radar_from_lidar=np.eye(4),
    sigma_xy=0.6,
    mahalanobis_thresh=3.0,
)

print(f"LiDAR points before removal: {lidar_points.shape[0]}")
print(f"LiDAR points after removal:  {keep_mask.sum()}")

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

ax = axes[0]
ax.scatter(lidar_points[:n_static, 0], lidar_points[:n_static, 1], s=8, c="gray", label="Static background")
ax.scatter(lidar_points[n_static:, 0], lidar_points[n_static:, 1], s=20, c="red", label="Moving car (in raw scan)")
ax.set_title("Before: raw LiDAR scan")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.legend()
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

ax = axes[1]
kept = lidar_points[keep_mask]
removed = lidar_points[~keep_mask]
ax.scatter(kept[:, 0], kept[:, 1], s=8, c="gray", label="Kept (static)")
ax.scatter(removed[:, 0], removed[:, 1], s=20, c="red", marker="x", label="Removed (flagged dynamic)")
ax.set_title("After: radar-guided dynamic removal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.legend()
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

fig.suptitle("Stage 2: Radar-guided Dynamic Point Removal")
fig.tight_layout()

out_path = Path(__file__).resolve().parents[1] / "outputs" / "dynamic_removal_demo.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved plot to {out_path}")

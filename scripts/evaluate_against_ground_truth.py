import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from garlio.evaluation import compute_ate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_real_data_pipeline import load_data, run_pipeline


def load_ground_truth(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    times = data[:, 0]
    pos = data[:, 1:4]
    return times, pos


def main():
    base = Path(__file__).resolve().parents[1]
    imu, radar, lidar = load_data(base / "real_data" / "extracted_data")
    gt_times, gt_pos = load_ground_truth(base / "real_data" / "ground_truth" / "ref_tls_T_xt32.csv")

    print(f"Loaded {len(imu)} IMU samples, {len(radar['frames'])} radar frames, {len(lidar['frames'])} lidar frames")
    print(f"Ground truth: {len(gt_times)} poses, span {gt_times[-1] - gt_times[0]:.1f}s")

    dead_reckoning, times_dr = run_pipeline(imu, radar, lidar, use_corrections=False)
    corrected, times_c = run_pipeline(imu, radar, lidar, use_corrections=True)

    result_dr = compute_ate(times_dr, dead_reckoning, gt_times, gt_pos, max_dt=0.06)
    result_c = compute_ate(times_c, corrected, gt_times, gt_pos, max_dt=0.06)

    print()
    print("=== ATE against real ground truth (after best-fit rigid alignment) ===")
    print(f"Dead-reckoning : n={result_dr['n_matched']}  RMSE={result_dr['rmse']:.3f} m  "
          f"mean={result_dr['mean']:.3f} m  median={result_dr['median']:.3f} m  max={result_dr['max']:.3f} m")
    print(f"Corrected      : n={result_c['n_matched']}  RMSE={result_c['rmse']:.3f} m  "
          f"mean={result_c['mean']:.3f} m  median={result_c['median']:.3f} m  max={result_c['max']:.3f} m")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    ax.plot(result_dr["matched_gt"][:, 0], result_dr["matched_gt"][:, 1], "k-", linewidth=2, label="Ground truth")
    ax.plot(result_dr["aligned_est"][:, 0], result_dr["aligned_est"][:, 1], "r--", linewidth=1.5, label="Dead-reckoning (aligned)")
    ax.set_title(f"IMU-only vs ground truth\nRMSE={result_dr['rmse']:.2f} m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(result_c["matched_gt"][:, 0], result_c["matched_gt"][:, 1], "k-", linewidth=2, label="Ground truth")
    ax.plot(result_c["aligned_est"][:, 0], result_c["aligned_est"][:, 1], "b--", linewidth=1.5, label="Corrected (aligned)")
    ax.set_title(f"Corrected vs ground truth\nRMSE={result_c['rmse']:.2f} m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Real Snail-Radar Sequence vs Ground Truth (best-fit rigid alignment)")
    fig.tight_layout()

    out_path = base / "outputs" / "ate_vs_ground_truth.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()

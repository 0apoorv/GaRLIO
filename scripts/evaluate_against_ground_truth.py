import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from garlio.evaluation import compute_ate, compute_relative_errors
from garlio.snail_radar_extrinsics import imu_poses_to_xt32

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_real_data_pipeline import load_data, run_pipeline


def load_ground_truth(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    times = data[:, 0]
    pos = data[:, 1:4]
    # Snail-Radar ref_tls_T_xt32.csv stores orientation as quaternion x,y,z,w.
    rotations = Rotation.from_quat(data[:, 4:8]).as_matrix()
    return times, pos, rotations


def main():
    parser = argparse.ArgumentParser(description="Evaluate GaRLIO against Snail-Radar ground truth")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Folder containing real_data/extracted_data and real_data/ground_truth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
        help="Directory for generated evaluation plots",
    )
    parser.add_argument(
        "--mat-output",
        type=Path,
        default=None,
        help="Path for the MATLAB evaluation input file (default: <output-dir>/trajectory_evaluation.mat)",
    )
    args = parser.parse_args()
    data_root = args.data_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    imu, radar, lidar = load_data(data_root / "real_data" / "extracted_data")
    gt_times, gt_pos, gt_rotations = load_ground_truth(data_root / "real_data" / "ground_truth" / "ref_tls_T_xt32.csv")

    print(f"Loaded {len(imu)} IMU samples, {len(radar['frames'])} radar frames, {len(lidar['frames'])} lidar frames")
    print(f"Ground truth: {len(gt_times)} poses, span {gt_times[-1] - gt_times[0]:.1f}s")

    dr_started = time.perf_counter()
    dead_reckoning, rotations_dr, times_dr = run_pipeline(
        imu, radar, lidar, use_corrections=False, return_rotations=True
    )
    dr_seconds = time.perf_counter() - dr_started
    corrected_started = time.perf_counter()
    corrected, rotations_c, times_c = run_pipeline(
        imu, radar, lidar, use_corrections=True, return_rotations=True
    )
    corrected_seconds = time.perf_counter() - corrected_started

    # The filter state is the X36D IMU pose; ref_tls_T_xt32.csv contains
    # XT32 poses. Convert both estimated trajectories into the XT32 frame.
    dead_reckoning, rotations_dr = imu_poses_to_xt32(dead_reckoning, rotations_dr)
    corrected, rotations_c = imu_poses_to_xt32(corrected, rotations_c)

    result_dr = compute_ate(times_dr, dead_reckoning, gt_times, gt_pos, max_dt=0.06)
    result_c = compute_ate(times_c, corrected, gt_times, gt_pos, max_dt=0.06)
    relative_dr = compute_relative_errors(
        times_dr, dead_reckoning, rotations_dr, gt_times, gt_pos, gt_rotations,
        interval_s=1.0, max_dt=0.06, interval_tolerance_s=0.06,
    )
    relative_c = compute_relative_errors(
        times_c, corrected, rotations_c, gt_times, gt_pos, gt_rotations,
        interval_s=1.0, max_dt=0.06, interval_tolerance_s=0.06,
    )

    mat_output = args.mat_output or (output_dir / "trajectory_evaluation.mat")
    mat_output.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        mat_output,
        {
            # MATLAB uses columns for samples and 3x3xN for rotation sequences.
            "times_dr": times_dr.reshape(1, -1),
            "positions_dr": dead_reckoning.T,
            "rotations_dr": np.transpose(rotations_dr, (1, 2, 0)),
            "times_corrected": times_c.reshape(1, -1),
            "positions_corrected": corrected.T,
            "rotations_corrected": np.transpose(rotations_c, (1, 2, 0)),
            "gt_times": gt_times.reshape(1, -1),
            "gt_positions": gt_pos.T,
            "gt_rotations": np.transpose(gt_rotations, (1, 2, 0)),
            "timestamp_tolerance_s": 0.06,
            "relative_interval_s": 1.0,
            "interval_tolerance_s": 0.06,
            "python_ate_dr_rmse": result_dr["rmse"],
            "python_ate_corrected_rmse": result_c["rmse"],
            # Per-pose translation APE after the same Umeyama alignment used by ATE.
            "ape_times_dr": result_dr["matched_times"].reshape(1, -1),
            "ape_errors_dr_m": result_dr["errors"].reshape(1, -1),
            "ape_times_corrected": result_c["matched_times"].reshape(1, -1),
            "ape_errors_corrected_m": result_c["errors"].reshape(1, -1),
            "python_rte_dr_rmse": relative_dr["translation_rmse"],
            "python_rte_corrected_rmse": relative_c["translation_rmse"],
            "python_rre_dr_rmse_rad": relative_dr["rotation_rmse_rad"],
            "python_rre_corrected_rmse_rad": relative_c["rotation_rmse_rad"],
            "python_ate_dr_n_matched": result_dr["n_matched"],
            "python_ate_corrected_n_matched": result_c["n_matched"],
            "python_relative_dr_n_pairs": relative_dr["n_pairs"],
            "python_relative_corrected_n_pairs": relative_c["n_pairs"],
        },
        do_compression=True,
    )
    print(f"Saved MATLAB input to {mat_output}")

    print()
    print("=== ATE against real ground truth (after best-fit rigid alignment) ===")
    print(f"Dead-reckoning : n={result_dr['n_matched']}  RMSE={result_dr['rmse']:.3f} m  "
          f"mean={result_dr['mean']:.3f} m  median={result_dr['median']:.3f} m  max={result_dr['max']:.3f} m")
    print(f"Corrected      : n={result_c['n_matched']}  RMSE={result_c['rmse']:.3f} m  "
          f"mean={result_c['mean']:.3f} m  median={result_c['median']:.3f} m  max={result_c['max']:.3f} m")
    print("APE is the per-pose aligned translation error; ATE RMSE summarizes that series.")
    print(f"Pipeline runtime: dead-reckoning={dr_seconds:.2f}s ({dr_seconds / max(1, len(times_dr) - 1):.3f}s/scan), "
          f"corrected={corrected_seconds:.2f}s ({corrected_seconds / max(1, len(times_c) - 1):.3f}s/scan)")
    print("\n=== Relative errors (1.0-second intervals; no global alignment) ===")
    print(f"Dead-reckoning : n={relative_dr['n_matched']} poses, {relative_dr['n_pairs']} pairs  "
          f"RTE RMSE={relative_dr['translation_rmse']:.3f} m  "
          f"RRE RMSE={relative_dr['rotation_rmse_deg']:.3f} deg "
          f"({relative_dr['rotation_rmse_rad']:.4f} rad)")
    print(f"Corrected      : n={relative_c['n_matched']} poses, {relative_c['n_pairs']} pairs  "
          f"RTE RMSE={relative_c['translation_rmse']:.3f} m  "
          f"RRE RMSE={relative_c['rotation_rmse_deg']:.3f} deg "
          f"({relative_c['rotation_rmse_rad']:.4f} rad)")

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

    out_path = output_dir / "ate_vs_ground_truth.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")

    fig_ape, ax_ape = plt.subplots(figsize=(12, 5))
    ax_ape.plot(result_dr["matched_times"] - times_dr[0], result_dr["errors"],
                color="tab:red", linewidth=1, label="IMU-only dead reckoning")
    ax_ape.plot(result_c["matched_times"] - times_c[0], result_c["errors"],
                color="tab:blue", linewidth=1, label="LiDAR/radar corrected")
    ax_ape.set_title("Absolute Pose Error (translation) by matched pose")
    ax_ape.set_xlabel("Time from trajectory start [s]")
    ax_ape.set_ylabel("Aligned position error [m]")
    ax_ape.grid(True, alpha=0.3)
    ax_ape.legend()
    fig_ape.tight_layout()
    ape_path = output_dir / "absolute_pose_error_vs_time.png"
    fig_ape.savefig(ape_path, dpi=150, bbox_inches="tight")
    print(f"Saved APE plot to {ape_path}")

    fig_rel, axes_rel = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for label, result, color in [
        ("Dead-reckoning", relative_dr, "tab:red"),
        ("Corrected", relative_c, "tab:blue"),
    ]:
        pair_midpoints = result["pair_times"].mean(axis=1) - times_dr[0]
        axes_rel[0].plot(
            pair_midpoints, result["translation_errors"], ".-",
            markersize=3, linewidth=1, color=color, label=label,
        )
        axes_rel[1].plot(
            pair_midpoints, result["rotation_errors_deg"], ".-",
            markersize=3, linewidth=1, color=color, label=label,
        )
    axes_rel[0].set_ylabel("RTE [m]")
    axes_rel[1].set_ylabel("RRE [deg]")
    axes_rel[1].set_xlabel("Time from trajectory start [s]")
    axes_rel[0].set_title("Relative errors over 1-second trajectory intervals")
    for ax in axes_rel:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig_rel.tight_layout()
    rel_path = output_dir / "relative_errors_vs_time.png"
    fig_rel.savefig(rel_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {rel_path}")


if __name__ == "__main__":
    main()

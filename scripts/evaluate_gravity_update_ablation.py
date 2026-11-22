"""Ablate radar Doppler and gravity-direction updates on the same sequence.

Both runs retain the same IMU propagation, LiDAR updates, radar Doppler
velocity updates, robust radar weighting, calibration, and evaluation settings.
Four runs separate radar Doppler and gravity-direction update effects.

Example:
    python scripts/evaluate_gravity_update_ablation.py \
        --data-root /path/to/garlio_btp_full
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_real_data_pipeline import load_data, run_pipeline
from evaluate_against_ground_truth import load_ground_truth
from garlio.evaluation import compute_ate, compute_relative_errors
from garlio.snail_radar_extrinsics import imu_poses_to_xt32


def frame_rpe(est_times, est_pos, est_rot, gt_times, gt_pos, gt_rot):
    """Compute consecutive-pose RPE after nearest-timestamp association."""
    matches = []
    last_gt_index = -1
    for est_index, timestamp in enumerate(est_times):
        gt_index = int(np.argmin(np.abs(gt_times - timestamp)))
        if abs(gt_times[gt_index] - timestamp) <= 0.06 and gt_index > last_gt_index:
            matches.append((est_index, gt_index))
            last_gt_index = gt_index
    if len(matches) < 2:
        raise ValueError("Fewer than two unique timestamp-matched poses for frame RPE")

    translation_errors = []
    rotation_errors = []
    for (i0, j0), (i1, j1) in zip(matches[:-1], matches[1:]):
        if i1 != i0 + 1 or j1 != j0 + 1:
            continue
        est_rel_R = est_rot[i0].T @ est_rot[i1]
        gt_rel_R = gt_rot[j0].T @ gt_rot[j1]
        rotation_error = gt_rel_R.T @ est_rel_R
        cos_angle = np.clip((np.trace(rotation_error) - 1.0) / 2.0, -1.0, 1.0)
        rotation_errors.append(float(np.arccos(cos_angle)))

        est_rel_t = est_rot[i0].T @ (est_pos[i1] - est_pos[i0])
        gt_rel_t = gt_rot[j0].T @ (gt_pos[j1] - gt_pos[j0])
        translation_errors.append(float(np.linalg.norm(est_rel_t - gt_rel_t)))

    if not translation_errors:
        raise ValueError("No consecutive frame pairs after timestamp association")
    return {
        "frame_rpe_translation_rmse_m": float(np.sqrt(np.mean(np.square(translation_errors)))),
        "frame_rpe_rotation_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(np.square(rotation_errors))))),
        "n_frame_matched": len(matches),
        "frame_translation_pairs": len(translation_errors),
        "frame_rotation_pairs": len(rotation_errors),
    }


def evaluate_variant(label, times, positions, rotations, gt_times, gt_pos, gt_rot):
    ate = compute_ate(times, positions, gt_times, gt_pos, max_dt=0.06)
    relative = compute_relative_errors(
        times, positions, rotations, gt_times, gt_pos, gt_rot,
        interval_s=1.0, max_dt=0.06, interval_tolerance_s=0.06,
    )
    frame = frame_rpe(times, positions, rotations, gt_times, gt_pos, gt_rot)
    return {
        "variant": label,
        "ate_rmse_m": ate["rmse"],
        "rte_1s_rmse_m": relative["translation_rmse"],
        "rre_1s_rmse_deg": relative["rotation_rmse_deg"],
        "frame_rpe_translation_rmse_m": frame["frame_rpe_translation_rmse_m"],
        "frame_rpe_rotation_rmse_deg": frame["frame_rpe_rotation_rmse_deg"],
        "ate_matched_poses": ate["n_matched"],
        "relative_pairs": relative["n_pairs"],
        "frame_matched_poses": frame["n_frame_matched"],
        "frame_translation_pairs": frame["frame_translation_pairs"],
        "frame_rotation_pairs": frame["frame_rotation_pairs"],
        "positions": positions,
        "rotations": rotations,
        "times": times,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs" / "velocity_gravity_ablation",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    imu, radar, lidar = load_data(args.data_root / "real_data" / "extracted_data")
    gt_times, gt_pos, gt_rot = load_ground_truth(
        args.data_root / "real_data" / "ground_truth" / "ref_tls_T_xt32.csv"
    )
    print(
        f"Loaded {len(imu)} IMU samples, {len(radar['frames'])} radar frames, "
        f"{len(lidar['frames'])} LiDAR frames, and {len(gt_times)} GT poses."
    )

    # Same random seed and settings. The first pair isolates radar Doppler
    # updates (gravity off); the latter pair isolates the added gravity update.
    runs = []
    for label, radar_enabled, gravity_enabled in (
        ("LiDAR only", False, False),
        ("LiDAR+radar, gravity OFF", True, False),
        ("LiDAR only, gravity ON", False, True),
        ("LiDAR+radar, gravity ON", True, True),
    ):
        print(f"\nRunning {label}...")
        pos, rot, times = run_pipeline(
            imu, radar, lidar,
            use_corrections=True,
            rng_seed=0,
            return_rotations=True,
            use_gravity_update=gravity_enabled,
            use_radar_update=radar_enabled,
        )
        pos, rot = imu_poses_to_xt32(pos, rot)
        runs.append(evaluate_variant(label, times, pos, rot, gt_times, gt_pos, gt_rot))

    metric_keys = [
        "ate_rmse_m", "rte_1s_rmse_m", "rre_1s_rmse_deg",
        "frame_rpe_translation_rmse_m", "frame_rpe_rotation_rmse_deg",
        "ate_matched_poses", "relative_pairs", "frame_matched_poses",
        "frame_translation_pairs", "frame_rotation_pairs",
    ]
    report_path = args.output_dir / "gravity_update_ablation.csv"
    with report_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", *metric_keys])
        writer.writeheader()
        for run in runs:
            writer.writerow({key: run[key] for key in ["variant", *metric_keys]})

    mat_data = {
        "gt_times": gt_times.reshape(1, -1),
        "gt_positions": gt_pos.T,
        "gt_rotations": np.transpose(gt_rot, (1, 2, 0)),
    }
    for suffix, run in zip(
        (
            "lidar_only", "radar_no_gravity", "lidar_with_gravity",
            "radar_with_gravity_on",
        ),
        runs,
    ):
        mat_data[f"times_{suffix}"] = run["times"].reshape(1, -1)
        mat_data[f"positions_{suffix}"] = run["positions"].T
        mat_data[f"rotations_{suffix}"] = np.transpose(run["rotations"], (1, 2, 0))
        for key in metric_keys:
            mat_data[f"{key}_{suffix}"] = run[key]
    mat_path = args.output_dir / "gravity_update_ablation.mat"
    savemat(mat_path, mat_data, do_compression=True)

    print("\nRMSE comparison (same sequence, timestamp matching, and evaluation):")
    for run in runs:
        print(f"\n{run['variant']}")
        for key in metric_keys[:5]:
            print(f"  {key:36s} {run[key]:.9f}")
    print(f"\nSaved {report_path}")
    print(f"Saved {mat_path}")


if __name__ == "__main__":
    main()

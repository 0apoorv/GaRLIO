"""Cross-check exported GaRLIO trajectory metrics with Michael Grupp's evo.

Run from the repository root:
    python scripts/compare_metrics_with_evo.py outputs/trajectory_evaluation.mat
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation

from evo.core import metrics, sync, units
from evo.core.trajectory import PoseTrajectory3D


def trajectory(times, positions, rotations, name):
    q_xyzw = Rotation.from_matrix(np.moveaxis(rotations, 2, 0)).as_quat()
    q_wxyz = q_xyzw[:, [3, 0, 1, 2]]
    return PoseTrajectory3D(
        positions_xyz=positions.T,
        orientations_quat_wxyz=q_wxyz,
        timestamps=times.reshape(-1),
        name=name,
    )


def summarize(metric, ref, est):
    metric.process_data((ref, est))
    stats = metric.get_all_statistics()
    return float(stats["rmse"]), len(metric.error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mat_file", nargs="?", type=Path,
        default=Path("outputs/trajectory_evaluation.mat"),
    )
    parser.add_argument("--max-dt", type=float, default=0.06)
    args = parser.parse_args()
    data = loadmat(args.mat_file)
    ref = trajectory(data["gt_times"], data["gt_positions"], data["gt_rotations"], "ground truth")

    for suffix, label in [("dr", "dead reckoning"), ("corrected", "corrected")]:
        est = trajectory(
            data[f"times_{suffix}"], data[f"positions_{suffix}"],
            data[f"rotations_{suffix}"], label,
        )
        ref_sync, est_sync = sync.associate_trajectories(ref, est, max_diff=args.max_dt)

        # APE/ATE uses the same SE(3) Umeyama alignment as our implementation.
        est_aligned = PoseTrajectory3D(
            positions_xyz=est_sync.positions_xyz.copy(),
            orientations_quat_wxyz=est_sync.orientations_quat_wxyz.copy(),
            timestamps=est_sync.timestamps.copy(),
            name=label,
        )
        est_aligned.align(ref_sync, correct_scale=False)
        ape, n_ape = summarize(
            metrics.APE(metrics.PoseRelation.translation_part), ref_sync, est_aligned
        )

        rpe_frame, n_frame = summarize(
            metrics.RPE(
                pose_relation=metrics.PoseRelation.translation_part,
                delta=1, delta_unit=units.Unit.frames,
            ), ref_sync, est_sync,
        )
        rpe_frame_rot, n_frame_rot = summarize(
            metrics.RPE(
                pose_relation=metrics.PoseRelation.rotation_angle_deg,
                delta=1, delta_unit=units.Unit.frames,
            ), ref_sync, est_sync,
        )

        print(f"{label}: matched poses={len(ref_sync.poses_se3)}")
        print(f"  evo APE translation RMSE / ATE: {ape:.9f} m (n={n_ape})")
        print(f"  evo frame-to-frame RPE:         {rpe_frame:.9f} m (n={n_frame})")
        print(f"  evo frame-to-frame rotation:    {rpe_frame_rot:.9f} deg (n={n_frame_rot})")
        print("  one-second RTE/RRE remain our fixed-time metrics (evo RPE uses frame/path/angle deltas).")


if __name__ == "__main__":
    main()

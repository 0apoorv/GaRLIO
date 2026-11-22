"""Synthetic regression tests for GaRLIO's ATE/RTE/RRE evaluation.

Run from the repository root with:
    PYTHONPATH=src python -m unittest discover -s tests -p "test_evaluation*.py"
"""
import unittest

import numpy as np

from garlio.evaluation import compute_relative_errors


def rot_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class TestRelativeErrors(unittest.TestCase):
    def setUp(self):
        self.times = np.array([0.0, 1.0, 2.0, 3.0])
        self.gt_pos = np.column_stack((self.times, np.zeros(4), np.zeros(4)))
        self.gt_rot = np.repeat(np.eye(3)[None, :, :], len(self.times), axis=0)
        self.kwargs = dict(interval_s=1.0, max_dt=1e-9, interval_tolerance_s=1e-9)

    def evaluate(self, est_pos=None, est_rot=None):
        return compute_relative_errors(
            self.times,
            self.gt_pos if est_pos is None else est_pos,
            self.gt_rot if est_rot is None else est_rot,
            self.times,
            self.gt_pos,
            self.gt_rot,
            **self.kwargs,
        )

    def test_identical_trajectories_have_zero_relative_error(self):
        result = self.evaluate()
        self.assertEqual(result["n_pairs"], 3)
        self.assertAlmostEqual(result["translation_rmse"], 0.0, places=12)
        self.assertAlmostEqual(result["rotation_rmse_deg"], 0.0, places=12)

    def test_constant_global_translation_does_not_change_relative_error(self):
        shifted = self.gt_pos + np.array([5.0, -2.0, 1.0])
        result = self.evaluate(est_pos=shifted)
        self.assertAlmostEqual(result["translation_rmse"], 0.0, places=12)
        self.assertAlmostEqual(result["rotation_rmse_deg"], 0.0, places=12)

    def test_known_translation_scale_error(self):
        # Each estimated 1-second displacement is 10% too long: error = 0.1 m.
        scaled = 1.1 * self.gt_pos
        result = self.evaluate(est_pos=scaled)
        self.assertAlmostEqual(result["translation_rmse"], 0.1, places=12)
        self.assertAlmostEqual(result["rotation_rmse_deg"], 0.0, places=12)

    def test_known_rotation_drift(self):
        angle_step = np.deg2rad(5.0)
        estimated_rotations = np.stack(
            [rot_z(i * angle_step) for i in range(len(self.times))]
        )
        # Zero translation isolates the relative rotation error.
        stationary = np.zeros_like(self.gt_pos)
        result = compute_relative_errors(
            self.times, stationary, estimated_rotations,
            self.times, stationary, self.gt_rot, **self.kwargs
        )
        self.assertAlmostEqual(result["translation_rmse"], 0.0, places=12)
        self.assertAlmostEqual(result["rotation_rmse_deg"], 5.0, places=10)


if __name__ == "__main__":
    unittest.main()

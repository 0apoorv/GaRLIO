"""20230920/data1 Snail-Radar sensor extrinsics.

Transforms were read from that sequence's ``tf_static.launch``.  Each matrix
``T_parent_child`` maps coordinates in the child frame into the parent frame.
The dataset defines ``base_link`` at the XT32 origin, with forward-left-up axes.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def _transform(translation, quaternion_xyzw):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    T[:3, 3] = np.asarray(translation, dtype=float)
    return T


def invert_transform(T):
    T = np.asarray(T, dtype=float)
    inverse = np.eye(4)
    inverse[:3, :3] = T[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ T[:3, 3]
    return inverse


T_BASE_X36D = _transform(
    [-0.06, 0.0, -0.16],
    [-0.0007904608601149109, -0.003950257535560073,
     -0.0007041813888545112, 0.9999916373478345],
)
T_BASE_OCULII = _transform(
    [0.07, 0.0, -0.115],
    [-0.013647489962849485, -0.004769404083372926,
     -0.009941686986431454, 0.9998460689837546],
)
T_BASE_XT32 = _transform(
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.7071067811865475, 0.7071067811865475],
)

# State poses are X36D IMU poses.  Measurement points are in XT32/Oculii.
T_X36D_XT32 = invert_transform(T_BASE_X36D) @ T_BASE_XT32
T_X36D_OCULII = invert_transform(T_BASE_X36D) @ T_BASE_OCULII


def imu_poses_to_xt32(positions, rotations):
    """Convert world-from-X36D poses to world-from-XT32 poses."""
    positions = np.asarray(positions, dtype=float)
    rotations = np.asarray(rotations, dtype=float)
    rotation_i_l = T_X36D_XT32[:3, :3]
    translation_i_l = T_X36D_XT32[:3, 3]
    xt32_rotations = rotations @ rotation_i_l
    xt32_positions = positions + np.einsum("nij,j->ni", rotations, translation_i_l)
    return xt32_positions, xt32_rotations

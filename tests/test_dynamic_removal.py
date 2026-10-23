import numpy as np

from garlio.dynamic_removal import remove_dynamic_lidar_points, transform_points


def identity_T():
    return np.eye(4)


def test_lidar_points_near_dynamic_radar_are_removed():
    lidar_points = np.array(
        [
            [5.0, 5.0, 0.0],
            [10.0, 10.0, 0.0],
            [-3.0, -3.0, 0.0],
        ]
    )
    dynamic_radar_points = np.array([[5.05, 4.98, 0.0]])

    mask = remove_dynamic_lidar_points(
        lidar_points, dynamic_radar_points, identity_T(), sigma_xy=0.3, mahalanobis_thresh=3.0
    )

    assert not mask[0]
    assert mask[1]
    assert mask[2]


def test_no_dynamic_points_keeps_everything():
    lidar_points = np.random.default_rng(0).uniform(-5, 5, size=(20, 3))
    dynamic_radar_points = np.empty((0, 3))

    mask = remove_dynamic_lidar_points(
        lidar_points, dynamic_radar_points, identity_T()
    )

    assert mask.all()


def test_far_lidar_points_are_untouched():
    lidar_points = np.array([[100.0, 100.0, 0.0], [-100.0, -100.0, 0.0]])
    dynamic_radar_points = np.array([[0.0, 0.0, 0.0]])

    mask = remove_dynamic_lidar_points(
        lidar_points, dynamic_radar_points, identity_T(), sigma_xy=0.5, mahalanobis_thresh=3.0
    )

    assert mask.all()


def test_transform_points_applies_rotation_and_translation():
    points = np.array([[1.0, 0.0, 0.0]])
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = np.array([10.0, 0.0, 0.0])

    transformed = transform_points(points, T)

    assert np.allclose(transformed, [[10.0, 1.0, 0.0]], atol=1e-10)


def test_removal_respects_extrinsic_transform_between_sensors():
    lidar_points = np.array([[1.0, 0.0, 0.0]])
    T_radar_from_lidar = np.eye(4)
    T_radar_from_lidar[:3, 3] = np.array([2.0, 0.0, 0.0])

    dynamic_radar_points = np.array([[3.0, 0.0, 0.0]])

    mask = remove_dynamic_lidar_points(
        lidar_points,
        dynamic_radar_points,
        T_radar_from_lidar,
        sigma_xy=0.3,
        mahalanobis_thresh=3.0,
    )

    assert not mask[0]


def test_multiple_dynamic_points_each_remove_their_neighbors():
    lidar_points = np.array(
        [
            [1.0, 1.0, 0.0],
            [8.0, 8.0, 0.0],
            [20.0, 20.0, 0.0],
        ]
    )
    dynamic_radar_points = np.array([[1.02, 0.99, 0.0], [8.0, 7.95, 0.0]])

    mask = remove_dynamic_lidar_points(
        lidar_points, dynamic_radar_points, identity_T(), sigma_xy=0.3, mahalanobis_thresh=3.0
    )

    assert not mask[0]
    assert not mask[1]
    assert mask[2]

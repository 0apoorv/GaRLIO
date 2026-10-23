import numpy as np

from garlio.mapping import LocalPlaneMap, fit_local_plane


def test_fit_local_plane_recovers_known_normal_and_centroid():
    rng = np.random.default_rng(0)
    pts = np.column_stack(
        [rng.uniform(-2, 2, 50), rng.uniform(-2, 2, 50), np.zeros(50)]
    )
    fit = fit_local_plane(pts)

    assert fit.is_planar
    assert np.allclose(np.abs(fit.normal), [0.0, 0.0, 1.0], atol=1e-6)
    assert np.allclose(fit.centroid, [0.0, 0.0, 0.0], atol=0.5)


def test_fit_local_plane_rejects_non_planar_cluster():
    rng = np.random.default_rng(1)
    pts = rng.uniform(-2, 2, size=(50, 3))
    fit = fit_local_plane(pts)
    assert not fit.is_planar


def test_fit_local_plane_tilted_plane():
    rng = np.random.default_rng(2)
    x = rng.uniform(-2, 2, 60)
    y = rng.uniform(-2, 2, 60)
    z = 0.5 * x
    pts = np.column_stack([x, y, z])
    fit = fit_local_plane(pts)

    assert fit.is_planar
    expected_normal = np.array([-0.5, 0.0, 1.0])
    expected_normal /= np.linalg.norm(expected_normal)
    assert (
        np.allclose(fit.normal, expected_normal, atol=1e-4)
        or np.allclose(fit.normal, -expected_normal, atol=1e-4)
    )


def test_local_plane_map_query_returns_plane_near_flat_region():
    rng = np.random.default_rng(3)
    floor = np.column_stack(
        [rng.uniform(-10, 10, 500), rng.uniform(-10, 10, 500), np.zeros(500)]
    )
    plane_map = LocalPlaneMap(floor, k_neighbors=10)

    fit = plane_map.query(np.array([1.0, 1.0, 0.05]))
    assert fit is not None
    assert np.allclose(np.abs(fit.normal), [0.0, 0.0, 1.0], atol=0.1)


def test_local_plane_map_rejects_query_in_sparse_noisy_region():
    rng = np.random.default_rng(4)
    scattered = rng.uniform(-10, 10, size=(2000, 3))
    plane_map = LocalPlaneMap(scattered, k_neighbors=30)

    fit = plane_map.query(np.array([0.0, 0.0, 0.0]))
    assert fit is None


def test_local_plane_map_two_walls_gives_different_normals():
    rng = np.random.default_rng(5)
    wall_x = np.column_stack(
        [np.full(200, 5.0), rng.uniform(-5, 5, 200), rng.uniform(0, 3, 200)]
    )
    wall_y = np.column_stack(
        [rng.uniform(-5, 5, 200), np.full(200, 5.0), rng.uniform(0, 3, 200)]
    )
    combined = np.vstack([wall_x, wall_y])
    plane_map = LocalPlaneMap(combined, k_neighbors=10)

    fit_x = plane_map.query(np.array([5.0, 0.0, 1.5]))
    fit_y = plane_map.query(np.array([0.0, 5.0, 1.5]))

    assert fit_x is not None and fit_y is not None
    assert np.allclose(np.abs(fit_x.normal), [1.0, 0.0, 0.0], atol=0.1)
    assert np.allclose(np.abs(fit_y.normal), [0.0, 1.0, 0.0], atol=0.1)

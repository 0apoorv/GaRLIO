import numpy as np


def hat(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    x, y, z = v
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])


def vee(m: np.ndarray) -> np.ndarray:
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def Exp(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = np.linalg.norm(phi)
    if theta < 1e-8:
        return np.eye(3) + hat(phi)
    axis_hat = hat(phi / theta)
    return (
        np.eye(3)
        + np.sin(theta) * axis_hat
        + (1.0 - np.cos(theta)) * (axis_hat @ axis_hat)
    )


def Log(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return vee(R - np.eye(3))
    return theta / (2.0 * np.sin(theta)) * vee(R - R.T)

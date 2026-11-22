import numpy as np


def align_umeyama(source: np.ndarray, target: np.ndarray):
    assert source.shape == target.shape
    n = source.shape[0]

    mu_src = source.mean(axis=0)
    mu_tgt = target.mean(axis=0)
    src_c = source - mu_src
    tgt_c = target - mu_tgt

    cov = (tgt_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    t = mu_tgt - R @ mu_src

    aligned_source = (R @ source.T).T + t
    return aligned_source, R, t


def match_by_time(est_times: np.ndarray, est_pos: np.ndarray, gt_times: np.ndarray, gt_pos: np.ndarray, max_dt: float = 0.05):
    matched_est = []
    matched_gt = []
    matched_times = []
    for i, t in enumerate(est_times):
        j = np.argmin(np.abs(gt_times - t))
        dt = abs(gt_times[j] - t)
        if dt <= max_dt:
            matched_est.append(est_pos[i])
            matched_gt.append(gt_pos[j])
            matched_times.append(t)
    return np.array(matched_est), np.array(matched_gt), np.array(matched_times)


def compute_ate(est_times: np.ndarray, est_pos: np.ndarray, gt_times: np.ndarray, gt_pos: np.ndarray, max_dt: float = 0.05):
    matched_est, matched_gt, matched_times = match_by_time(est_times, est_pos, gt_times, gt_pos, max_dt)
    if len(matched_est) < 3:
        raise ValueError(f"Only {len(matched_est)} matched points (need >=3 for alignment)")

    aligned_est, R, t = align_umeyama(matched_est, matched_gt)
    errors = np.linalg.norm(aligned_est - matched_gt, axis=1)

    return {
        "n_matched": len(matched_est),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "max": float(np.max(errors)),
        "aligned_est": aligned_est,
        "matched_gt": matched_gt,
        "matched_times": matched_times,
        "errors": errors,
        "R": R,
        "t": t,
    }


def compute_relative_errors(
    est_times: np.ndarray,
    est_pos: np.ndarray,
    est_rotations: np.ndarray,
    gt_times: np.ndarray,
    gt_pos: np.ndarray,
    gt_rotations: np.ndarray,
    interval_s: float,
    max_dt: float = 0.05,
    interval_tolerance_s: float = 0.05,
):
    """Compute fixed-time RTE and RRE from matched body-to-world poses.

    For each matched timestamp, the endpoint is the matched pose nearest
    ``interval_s`` later. A pair is included only when its actual duration is
    within ``interval_tolerance_s`` of the requested interval. Relative
    translations are expressed in the start pose's body frame; the returned
    translation error is the norm of the relative-transform error translation.
    Rotational errors are the geodesic angle in radians and degrees.

    No global alignment is applied: relative transforms are invariant to a
    common rigid transform of each trajectory when rotations and positions
    use the same world frame.
    """
    est_times = np.asarray(est_times, dtype=float).reshape(-1)
    gt_times = np.asarray(gt_times, dtype=float).reshape(-1)
    est_pos = np.asarray(est_pos, dtype=float)
    gt_pos = np.asarray(gt_pos, dtype=float)
    est_rotations = np.asarray(est_rotations, dtype=float)
    gt_rotations = np.asarray(gt_rotations, dtype=float)

    if interval_s <= 0:
        raise ValueError("interval_s must be positive")
    if max_dt < 0 or interval_tolerance_s < 0:
        raise ValueError("timestamp tolerances must be non-negative")
    if est_pos.shape != (len(est_times), 3) or gt_pos.shape != (len(gt_times), 3):
        raise ValueError("positions must have shape (number of timestamps, 3)")
    if est_rotations.shape != (len(est_times), 3, 3) or gt_rotations.shape != (len(gt_times), 3, 3):
        raise ValueError("rotations must have shape (number of timestamps, 3, 3)")
    if len(gt_times) == 0:
        raise ValueError("ground-truth timestamps must not be empty")

    # Match each estimate independently, following the ATE matcher’s nearest-
    # timestamp policy and tolerance. Retain source indices for orientations.
    matched = []
    for i, t in enumerate(est_times):
        j = int(np.argmin(np.abs(gt_times - t)))
        dt = abs(gt_times[j] - t)
        if dt <= max_dt:
            matched.append((i, j, t))

    translation_errors = []
    rotation_errors = []
    pair_times = []
    pair_durations = []
    for start in range(len(matched)):
        i, gi, t_start = matched[start]
        target_time = t_start + interval_s
        candidates = [
            k for k in range(start + 1, len(matched))
            if matched[k][2] > t_start
        ]
        if not candidates:
            continue
        end = min(candidates, key=lambda k: abs(matched[k][2] - target_time))
        j, gj, t_end = matched[end]
        duration = t_end - t_start
        if abs(duration - interval_s) > interval_tolerance_s:
            continue

        # T_WB maps body-frame coordinates into world coordinates.
        R_gt = gt_rotations[gi].T @ gt_rotations[gj]
        R_est = est_rotations[i].T @ est_rotations[j]
        t_gt = gt_rotations[gi].T @ (gt_pos[gj] - gt_pos[gi])
        t_est = est_rotations[i].T @ (est_pos[j] - est_pos[i])

        # inv(Delta_T_gt) @ Delta_T_est: rotation and translation residuals.
        R_error = R_gt.T @ R_est
        t_error = R_gt.T @ (t_est - t_gt)
        cos_angle = np.clip((np.trace(R_error) - 1.0) / 2.0, -1.0, 1.0)
        sin_angle = 0.5 * np.linalg.norm(
            np.array([
                R_error[2, 1] - R_error[1, 2],
                R_error[0, 2] - R_error[2, 0],
                R_error[1, 0] - R_error[0, 1],
            ])
        )
        translation_errors.append(float(np.linalg.norm(t_error)))
        rotation_errors.append(float(np.arctan2(sin_angle, cos_angle)))
        pair_times.append((t_start, t_end))
        pair_durations.append(duration)

    if not translation_errors:
        raise ValueError(
            "No valid pose pairs for the requested interval "
            f"({interval_s:g}s, tolerance {interval_tolerance_s:g}s); "
            f"{len(matched)} poses matched within {max_dt:g}s"
        )

    translation_errors = np.asarray(translation_errors)
    rotation_errors = np.asarray(rotation_errors)
    return {
        "n_matched": len(matched),
        "n_pairs": len(translation_errors),
        "interval_s": float(interval_s),
        "translation_errors": translation_errors,
        "translation_rmse": float(np.sqrt(np.mean(translation_errors ** 2))),
        "translation_mean": float(np.mean(translation_errors)),
        "translation_median": float(np.median(translation_errors)),
        "translation_max": float(np.max(translation_errors)),
        "rotation_errors_rad": rotation_errors,
        "rotation_errors_deg": np.rad2deg(rotation_errors),
        "rotation_rmse_rad": float(np.sqrt(np.mean(rotation_errors ** 2))),
        "rotation_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(rotation_errors ** 2)))),
        "rotation_mean_rad": float(np.mean(rotation_errors)),
        "rotation_mean_deg": float(np.rad2deg(np.mean(rotation_errors))),
        "pair_times": np.asarray(pair_times),
        "pair_durations": np.asarray(pair_durations),
    }

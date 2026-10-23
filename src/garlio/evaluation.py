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

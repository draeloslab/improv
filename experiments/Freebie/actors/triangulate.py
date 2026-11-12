"""Confidence-weighted DLT triangulation (the Pose2Sim / "weighted DLT" formulation) for an aniposelib CameraGroup.

aniposelib's CameraGroup.triangulate is a plain DLT: every camera that kept a keypoint pulls on it equally, so a view
the model was 0.31 sure of counts as much as one it was 0.99 sure of. Here each camera's two DLT rows are scaled by
its confidence before the SVD, so a weak view still contributes but cannot drag the point. With equal weights the
result is the same as aniposelib's (checked to < 1e-6 mm).

    p3 = weighted_dlt(cgroup, points_2d, weights)     # (C, N, 2) px, (C, N) or None -> (N, 3) mm

Vectorised over points (one batched SVD): ~0.1 ms for 42 keypoints x 4 cameras, no jax/jit warm-up.
"""
import numpy as np


def weighted_dlt(cgroup, points_2d, weights=None, min_views=2):
    """points_2d: (C, N, 2) distorted pixels, NaN = not seen. weights: (C, N) >= 0 (None = all 1).
    Returns (N, 3); NaN where fewer than min_views cameras have a finite point with weight > 0."""
    p = np.asarray(points_2d, dtype=float)
    C, N = p.shape[:2]
    und = np.empty_like(p)
    for c, cam in enumerate(cgroup.cameras):
        und[c] = cam.undistort_points(np.ascontiguousarray(p[c]))      # normalised image coordinates
    w = np.ones((C, N)) if weights is None else np.asarray(weights, dtype=float).copy()
    valid = np.isfinite(und).all(axis=2) & np.isfinite(w) & (w > 0)
    w = np.where(valid, w, 0.0)
    und = np.where(valid[..., None], und, 0.0)

    P = np.stack([cam.get_extrinsics_mat()[:3] for cam in cgroup.cameras])     # (C, 3, 4)
    # rows x * P[2] - P[0] and y * P[2] - P[1] per camera and point -> (N, C, 2, 4)
    A = und.transpose(1, 0, 2)[..., None] * P[None, :, 2:3, :] - P[None, :, 0:2, :]
    A = A * w.T[:, :, None, None]
    _, _, vh = np.linalg.svd(A.reshape(N, 2 * C, 4))
    X = vh[:, -1]
    with np.errstate(invalid='ignore', divide='ignore'):
        out = X[:, :3] / X[:, 3:4]
    out[valid.sum(axis=0) < min_views] = np.nan
    return out


def triangulator(cgroup, method='weighted_dlt', min_weight=0.0):
    """fn(points_2d (C, N, 2), weights (C, N) or None) -> (N, 3) for the chosen method.
    'weighted_dlt': weights clipped below at min_weight (so a kept but unconfident view still counts a little);
    'dlt': every view equal, weights ignored (aniposelib's unweighted DLT, computed here: ~2x faster)."""
    if method == 'dlt':
        # equal weights: the same solve as aniposelib's CameraGroup.triangulate (checked to < 1e-5 mm), without its jax call
        return lambda p, w=None: weighted_dlt(cgroup, p, None)
    if method != 'weighted_dlt':
        raise ValueError(f"unknown triangulation method {method!r}, expected 'weighted_dlt' or 'dlt'")

    def fn(p, w=None):
        """Weighted DLT, weights floored at min_weight where a point is present."""
        if w is not None and min_weight > 0:
            w = np.where(np.isfinite(w), np.maximum(w, min_weight), w)
        return weighted_dlt(cgroup, p, w)
    return fn


def reprojection_refine(a2, cg, rows, tri, reject_px, block=4000):
    """Re-triangulate every keypoint from its associated views, dropping the worst view while it is more than reject_px
    off and >= 3 views remain; a point still off with 2 views is rejected. Vectorised over frames.
    Returns points (T, 42, 3), views used (T, 42), reprojection error of the kept views (T, 42, n_calib), rejected (T, 42)
    and dropped (T, 42, n_calib): views that were offered but removed (a dropped view, or every view of a rejected point)."""
    T, C = a2.shape[:2]
    n_cal = len(cg.cameras)
    out = np.full((T, 42, 3), np.nan)
    nviews = np.zeros((T, 42), np.int8)
    err_all = np.full((T, 42, n_cal), np.nan, np.float32)
    rejected = np.zeros((T, 42), bool)
    dropped = np.zeros((T, 42, n_cal), bool)
    for b0 in range(0, T, block):
        b1 = min(T, b0 + block)
        n = (b1 - b0) * 42
        pts = np.full((n_cal, n, 2), np.nan)
        w = np.zeros((n_cal, n))
        for s in range(C):
            blk = a2[b0:b1, s].reshape(n, 3)
            pts[rows[s]] = blk[:, :2]
            w[rows[s]] = np.where(np.isfinite(blk[:, 0]), np.nan_to_num(blk[:, 2], nan=0.0), 0.0)
        use = np.isfinite(pts[..., 0]) & (w > 0)
        use0 = use.copy()
        for it in range(n_cal + 1):                 # solve, check; drop each failing point's worst view; solve again
            X = tri(np.where(use[..., None], pts, np.nan), np.where(use, w, 0.0))
            solved = np.isfinite(X[:, 0])
            proj = cg.project(np.where(solved[:, None], X, 0.0)).reshape(n_cal, n, 2)
            e = np.where(use & solved[None], np.linalg.norm(proj - pts, axis=2), np.nan)
            worst = np.max(np.where(np.isfinite(e), e, -1.0), axis=0)
            k = use.sum(0)
            drop = solved & (worst > reject_px) & (k >= 3)
            if not drop.any() or it == n_cal:
                break
            idx = np.flatnonzero(drop)
            use[np.argmax(np.where(np.isfinite(e[:, idx]), e[:, idx], -1.0), axis=0), idx] = False
        bad = solved & (worst > reject_px)          # still off with only 2 views left: no trustworthy point
        rejected[b0:b1] = bad.reshape(-1, 42)
        X[bad] = np.nan
        keep = solved & ~bad
        dropped[b0:b1] = (use0 & ~(use & keep[None])).T.reshape(-1, 42, n_cal)
        out[b0:b1] = X.reshape(-1, 42, 3)
        nviews[b0:b1] = np.where(keep, k, 0).reshape(-1, 42)
        err_all[b0:b1] = np.where(keep[None], e, np.nan).T.reshape(-1, 42, n_cal)
    return out, nviews, err_all, rejected, dropped

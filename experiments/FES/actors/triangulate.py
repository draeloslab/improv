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
    'dlt': aniposelib's unweighted CameraGroup.triangulate (weights ignored)."""
    if method == 'dlt':
        return lambda p, w=None: np.asarray(cgroup.triangulate(p, progress=False), dtype=float)
    if method != 'weighted_dlt':
        raise ValueError(f"unknown triangulation method {method!r}, expected 'weighted_dlt' or 'dlt'")

    def fn(p, w=None):
        """Weighted DLT, weights floored at min_weight where a point is present."""
        if w is not None and min_weight > 0:
            w = np.where(np.isfinite(w), np.maximum(w, min_weight), w)
        return weighted_dlt(cgroup, p, w)
    return fn

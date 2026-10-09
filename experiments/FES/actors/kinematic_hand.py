"""Kinematic hand fit: replace a triangulated 21-keypoint hand by the closest hand the skeleton can actually make.

The hand is a rigid palm (wrist, thumb CMC and the four MCPs at fixed places in the hand's own frame) plus five
chains of three bones with FIXED lengths. Each chain has a 2-degree-of-freedom base joint (flexion + abduction: the
finger MCPs, the thumb CMC) and two flexion joints (PIP / DIP; thumb MCP / IP), all within joint limits. With the
wrist's position and orientation that is 26 numbers per frame, found by robust least squares against the measured 3D
keypoints (missing ones simply drop out), warm-started from the previous frame, with a small pull towards the
previous joint angles. The output keeps every bone at its true length, never bends a joint backwards past its limit
and fills keypoints the cameras lost from the rest of the hand.

The palm shape, bone lengths and limits come from a hand model file (build it once with scripts/build_hand_model.py
from runs you trust; it stays valid for any later session whose calibration is metric).

    fit = KinematicHand(model['right'])
    points, info = fit.step(points_3d[:21])        # (21, 3) mm, NaN where unseen -> fitted (21, 3), dict
"""
import time

import numpy as np

FINGERS = {'thumb': [1, 2, 3, 4], 'index': [5, 6, 7, 8], 'middle': [9, 10, 11, 12],
           'ring': [13, 14, 15, 16], 'pinky': [17, 18, 19, 20]}
PALM = [0, 1, 5, 9, 13, 17]                 # wrist, thumb CMC, index/middle/ring/pinky MCP
# degrees, flexion positive towards the palm: [abduction, base flexion, middle flexion, distal flexion]
DEFAULT_LIMITS = {
    'thumb': [[-60, 60], [-60, 60], [-30, 90], [-30, 100]],
    'finger': [[-30, 30], [-30, 100], [-10, 120], [-15, 100]],
}


def hand_frame(p):
    """(…, 21, 3) hands -> rotation (…, 3, 3) whose columns are the hand axes: x wrist -> middle MCP, z normal to the
    palm (index MCP x pinky MCP side), y = z x x. The palmar sign of z is applied separately (model 'palmar_sign')."""
    x = p[..., 9, :] - p[..., 0, :]
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.cross(p[..., 5, :] - p[..., 0, :], p[..., 17, :] - p[..., 0, :])
    z = n - (n * x).sum(-1, keepdims=True) * x
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


def rotvec_to_matrix(r):
    """Rodrigues, batched: (B, 3) -> (B, 3, 3)."""
    th = np.linalg.norm(r, axis=-1, keepdims=True)
    k = r / np.where(th > 1e-12, th, 1.0)
    K = np.zeros(r.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = k[..., 2], -k[..., 1], k[..., 0]
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def matrix_to_rotvec(R):
    """(3, 3) -> (3,) rotation vector."""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-8:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - th < 1e-4:                      # near 180 deg: axis from the symmetric part
        w, V = np.linalg.eigh((R + R.T) / 2)
        return V[:, -1] * th
    return v / (2 * np.sin(th)) * th


class KinematicHand:
    """Fits one hand. `model`: dict with 'palm' {kp: [x, y, z] mm in the hand frame (z palmar)}, 'bones' {finger:
    [3 lengths mm]}, optionally 'limits_deg' {finger: [[lo, hi] x 4]}."""

    def __init__(self, model, temporal_weight=2.0, f_scale_mm=5.0, max_rms_mm=15.0, min_keypoints=8, max_iter=15):
        """temporal_weight: mm of residual per radian of joint-angle change since the last frame (0 = none).
        f_scale_mm: residual where the robust loss stops being quadratic. max_rms_mm: a fit worse than this is
        reported as failed. min_keypoints: measured keypoints needed to fit at all."""
        self.palm = np.array([model['palm'][str(k)] if str(k) in model['palm'] else model['palm'][k] for k in PALM], float)
        self.bones = {f: np.asarray(model['bones'][f], float) for f in FINGERS}
        limits = model.get('limits_deg') or {}
        lo, hi = [-np.inf] * 6, [np.inf] * 6
        for f in FINGERS:
            lim = limits.get(f, DEFAULT_LIMITS['thumb' if f == 'thumb' else 'finger'])
            lo += [np.radians(a) for a, _ in lim]; hi += [np.radians(b) for _, b in lim]
        self.lo, self.hi = np.array(lo), np.array(hi)
        self.temporal_weight, self.f_scale, self.max_rms = temporal_weight, f_scale_mm, max_rms_mm
        self.min_kp, self.max_iter = min_keypoints, max_iter
        # per finger: base point and a local frame [rest direction, lateral, palmar] in the hand frame
        self.base, self.F = {}, {}
        z = np.array([0.0, 0.0, 1.0])
        for f, idx in FINGERS.items():
            b = self.palm[PALM.index(idx[0])]
            d0 = b - self.palm[0] if f != 'thumb' else b - self.palm[0]
            d0 = d0 - (d0 @ z) * z if f != 'thumb' else d0       # fingers: metacarpal direction within the palm plane
            d0 /= np.linalg.norm(d0)
            e3 = z - (z @ d0) * d0; e3 /= np.linalg.norm(e3)
            self.base[f], self.F[f] = b, np.stack([d0, np.cross(e3, d0), e3], axis=1)
        self.prev = None

    # ------------------------------------------------------------------ model
    def forward(self, theta):
        """(B, 26) parameters -> (B, 21, 3) world points. theta = [wrist xyz, rotvec, per finger abd, flex x3]."""
        theta = np.atleast_2d(theta)
        B = len(theta)
        hand = np.zeros((B, 21, 3))
        for i, k in enumerate(PALM):
            hand[:, k] = self.palm[i]
        for j, (f, idx) in enumerate(FINGERS.items()):
            abd, f1, f2, f3 = (theta[:, 6 + 4 * j + m] for m in range(4))
            # direction after abduction then base flexion, in the finger's local frame
            c1 = np.cos(f1)
            d1 = np.stack([c1 * np.cos(abd), c1 * np.sin(abd), np.sin(f1)], -1)
            # later joints flex about the same (abducted) lateral axis: total flexion f1+f2(+f3) in that plane
            def along(phi):
                c = np.cos(phi)
                return np.stack([c * np.cos(abd), c * np.sin(abd), np.sin(phi)], -1)
            d2, d3 = along(f1 + f2), along(f1 + f2 + f3)
            L = self.bones[f]
            p0 = self.base[f]
            p1 = p0 + L[0] * d1 @ self.F[f].T
            p2 = p1 + L[1] * d2 @ self.F[f].T
            p3 = p2 + L[2] * d3 @ self.F[f].T
            hand[:, idx[1]], hand[:, idx[2]], hand[:, idx[3]] = p1, p2, p3
        R = rotvec_to_matrix(theta[:, 3:6])
        return theta[:, None, 0:3] + hand @ R.transpose(0, 2, 1)

    # -------------------------------------------------------------------- fit
    def _init(self, obs):
        """Wrist pose from the palm keypoints (Kabsch), joints straight."""
        ok = [i for i, k in enumerate(PALM) if np.isfinite(obs[k]).all()]
        th = np.zeros(26)
        if len(ok) < 3:
            return None
        A, Bm = self.palm[ok], obs[[PALM[i] for i in ok]]
        ca, cb = A.mean(0), Bm.mean(0)
        U, _, Vt = np.linalg.svd((A - ca).T @ (Bm - cb))
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        th[0:3], th[3:6] = cb - R @ ca, matrix_to_rotvec(R)
        th[6:] = np.clip(0.0, self.lo[6:], self.hi[6:])
        return th

    def _resid_batch(self, thetas, obs, mask):
        """Residuals (B, 3m [+20]) of B parameter sets: point errors (mm), then the temporal pull on the joints."""
        r = (self.forward(thetas)[:, mask] - obs[mask]).reshape(len(thetas), -1)
        if self.prev is not None and self.temporal_weight > 0:
            r = np.concatenate([r, self.temporal_weight * (thetas[:, 6:] - self.prev[6:])], axis=1)
        return r

    def _weights(self, r, m):
        """IRLS weights of the soft-L1 loss, per keypoint (its 3 residuals share one) and 1 for the prior terms."""
        e = np.linalg.norm(r[:3 * m].reshape(m, 3), axis=1)
        w = np.ones_like(r)
        w[:3 * m] = np.repeat(1.0 / np.sqrt(1.0 + (e / self.f_scale) ** 2), 3)
        return w

    def _cost(self, r, m):
        e2 = (r[:3 * m].reshape(m, 3) ** 2).sum(1)
        return 2 * self.f_scale ** 2 * np.sum(np.sqrt(1 + e2 / self.f_scale ** 2) - 1) + np.sum(r[3 * m:] ** 2)

    def _solve(self, x, obs, mask):
        """Levenberg-Marquardt with soft-L1 reweighting; joint limits by projection. Jacobian by forward
        differences, all 26 perturbations in one batched forward pass. Returns (x, iterations)."""
        m, h, lam = int(mask.sum()), 1e-5, 1e-3
        T = np.empty((27, 26))
        for it in range(1, self.max_iter + 1):
            T[:] = x; T[1:] += np.eye(26) * h
            R = self._resid_batch(T, obs, mask)
            r = R[0]; J = (R[1:] - r).T / h
            w = self._weights(r, m)
            A = J.T @ (w[:, None] * J); g = J.T @ (w * r)
            cost = self._cost(r, m)
            while True:
                step = np.linalg.solve(A + lam * (np.diag(np.diag(A)) + 1e-9 * np.eye(26)), -g)
                xn = np.clip(x + step, self.lo, self.hi)
                cn = self._cost(self._resid_batch(xn[None], obs, mask)[0], m)
                if cn < cost or lam > 1e6:
                    break
                lam *= 4
            if cn >= cost:
                break
            x, lam = xn, max(lam / 3, 1e-7)
            if cost - cn < 1e-4 * cost or np.abs(step).max() < 1e-5:
                break
        return x, it

    def step(self, obs):
        """obs: (21, 3) mm, NaN where not measured. Returns (fitted (21, 3) or None if the fit is worse than
        max_rms_mm, info dict)."""
        t0 = time.perf_counter()
        mask = np.isfinite(obs).all(1)
        if mask.sum() < self.min_kp:
            self.prev = None
            return None, dict(ok=False, reason='too few keypoints', n=int(mask.sum()))
        x0 = self.prev if self.prev is not None else self._init(obs)
        if x0 is None:
            return None, dict(ok=False, reason='no palm', n=int(mask.sum()))
        if self.prev is not None and np.linalg.norm(self.forward(self.prev)[0, 0] - obs[0]) > 60:
            x0 = self._init(obs) if np.isfinite(obs[0]).all() else x0     # the hand jumped: start from its palm
            if x0 is None:
                return None, dict(ok=False, reason='no palm', n=int(mask.sum()))
        x, iters = self._solve(np.clip(x0, self.lo, self.hi), obs, mask)
        fitted = self.forward(x)[0]
        err = np.linalg.norm(fitted[mask] - obs[mask], axis=1)
        rms = float(np.sqrt(np.mean(err ** 2)))
        ok = rms <= self.max_rms
        self.prev = x                                    # warm start next frame either way
        return (fitted if ok else None), dict(ok=ok, rms_mm=rms, n=int(mask.sum()), iters=iters,
                                              ms=(time.perf_counter() - t0) * 1e3, theta=x)


def build_model(hands, min_frames=50):
    """Hand model from many complete 3D hands (N, 21, 3) of ONE hand: median palm shape in the hand frame, median
    bone lengths, palmar direction from where the fingertips go. Returns (model dict, report dict)."""
    H = hands[np.isfinite(hands).all(axis=(1, 2))]
    if len(H) < min_frames:
        raise ValueError(f"only {len(H)} complete hands (need {min_frames})")
    R = hand_frame(H)                                                   # (N, 3, 3)
    local = np.einsum('nij,nkj->nki', R.transpose(0, 2, 1), H - H[:, :1])  # hand-frame coordinates
    # palmar side: fingers curl towards the palm, so the fingertips sit on the palmar side of the palm plane on average
    tips = local[:, [8, 12, 16, 20], 2].mean()
    sign = 1.0 if tips > 0 else -1.0
    local[..., 2] *= sign
    local[..., 1] *= sign                                               # keep the frame right-handed
    palm = {str(k): np.median(local[:, k], axis=0).round(2).tolist() for k in PALM}
    bones, cv = {}, {}
    for f, idx in FINGERS.items():
        L = np.stack([np.linalg.norm(H[:, idx[m + 1]] - H[:, idx[m]], axis=1) for m in range(3)], 1)
        bones[f] = np.median(L, 0).round(2).tolist()
        cv[f] = (np.std(L, 0) / np.mean(L, 0)).round(3).tolist()
    return (dict(palm=palm, bones=bones, limits_deg={'thumb': DEFAULT_LIMITS['thumb'],
                                                     **{f: DEFAULT_LIMITS['finger'] for f in FINGERS if f != 'thumb'}}),
            dict(frames=len(H), palmar_sign=sign, bone_cv=cv))


def mirror_model(model):
    """The other hand's model: the same bones, the palm reflected across the hand's x-z plane (thumb on the other side)."""
    out = dict(model)
    out['palm'] = {k: [v[0], -v[1], v[2]] for k, v in model['palm'].items()}
    return out

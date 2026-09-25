"""3D of ONE hand that stays in a known place (e.g. the stimulated hand an experimenter holds), from a
two-hand DLC model, frame by frame. `hand_association: region` in ProcessorBatch3D.

Why not hand_assoc3d.py: DLC gives a keypoint-level likelihood and its right/left labels are unreliable
across cameras, and the held hand is partly hidden, so whole-hand matching fails. Per step:
  1. each camera's two DLC hands: every combination (one hand per camera) is triangulated, keypoints need
     likelihood >= `likelihood` and must reproject within `max_px` in every camera that sees them;
  2. keep the combination whose hand sits where the target hand lives: 3D centre within `radius_mm` of its
     place AND, in every camera, 2D centre within `radius_px` of its usual image position (with only two
     cameras a wrong match can reproject perfectly, e.g. the monkey's chin matched to the hand);
  3. causal One-Euro filter per keypoint (missing keypoints keep their state; reset after `max_gap` frames).
The place is learned from the first `warmup_s` seconds (both hands clustered, numbered left -> right in
camera `order_camera`'s image, `hand` picks one) and then fixed. Nothing is output during warm-up.

Validated offline (analysis/stim_1153/pseudo_live.py): 17.8 ms/step with DLC on 2 cameras, MCP angles
within 1.1-1.5 deg of the whole-recording analysis.
"""
import itertools
import logging

import numpy as np

logger = logging.getLogger(__name__)


class OneEuroKeypoints:
    def __init__(self, n=21, min_cutoff=1.0, beta=0.05, d_cutoff=1.0, max_gap=5, fps=30.0):
        self.mc, self.beta, self.dc, self.max_gap, self.fps = min_cutoff, beta, d_cutoff, max_gap, fps
        self.x = np.full((n, 3), np.nan)
        self.dx = np.zeros((n, 3))
        self.gap = np.zeros(n, int)

    def step(self, z):
        a = lambda fc: 1.0 / (1.0 + self.fps / (2 * np.pi * fc))
        have = np.isfinite(z[:, 0])
        self.gap = np.where(have, 0, self.gap + 1)
        new = have & ~np.isfinite(self.x[:, 0])
        self.x[new], self.dx[new] = z[new], 0
        upd = have & ~new
        self.dx[upd] += a(self.dc) * ((z[upd] - self.x[upd]) * self.fps - self.dx[upd])
        self.x[upd] += a(self.mc + self.beta * np.abs(self.dx[upd])) * (z[upd] - self.x[upd])
        self.x[self.gap > self.max_gap] = np.nan
        return np.where(have[:, None], self.x, np.nan)


class HeldHandTracker:
    def __init__(self, cgroup, likelihood=0.3, max_px=15.0, warmup_s=10.0, hand=1, order_row=None,
                 radius_mm=80.0, radius_px=40.0, smooth=None, fps=30.0):
        self.cg = cgroup
        self.nrow = len(cgroup.cameras)
        self.lik, self.max_px, self.warmup_s, self.hand = likelihood, max_px, warmup_s, hand
        self.order_row = order_row
        self.radius_mm, self.radius_px = radius_mm, radius_px
        self.filter = OneEuroKeypoints(fps=fps, **(smooth or {}))
        self.t0 = None
        self.warm = []
        self.centre = self.home = None

    def _candidates(self, views):
        """views: {calib_row: (2 * 21, 3) x, y, likelihood}. Yields (3D (21, 3), used-rows)."""
        rows = list(views)
        for pick in itertools.product((0, 21), repeat=len(rows)):
            pts = np.full((self.nrow, 21, 2), np.nan)
            for r, off in zip(rows, pick):
                h = views[r][off:off + 21]
                ok = h[:, 2] >= self.lik
                pts[r][ok] = h[ok, :2]
            seen = np.isfinite(pts[..., 0]).sum(0) >= 2
            if seen.sum() < 3:
                continue
            pts[:, ~seen] = np.nan
            q = self.cg.triangulate(pts, progress=False)
            e = np.linalg.norm(self.cg.project(q) - pts, axis=-1)
            ok = np.nanmax(np.where(np.isfinite(e), e, 0), axis=0) <= self.max_px
            ok &= seen
            if ok.sum() >= 3:
                yield np.where(ok[:, None], q, np.nan), rows

    def _centre2d(self, q):
        return {r: np.nanmedian(self.cg.project(q)[r], 0) for r in range(self.nrow)}

    def step(self, t, views):
        """t: capture time (s); views: {calib_row: (42, 3)}. Returns (21, 3) mm or None."""
        if self.t0 is None:
            self.t0 = t
        cands = list(self._candidates(views)) if len(views) >= 2 else []
        if self.centre is None:
            self.warm += [q for q, _ in cands]
            if t - self.t0 >= self.warmup_s and len(self.warm) >= 20:
                self._learn()
            return None
        best = None
        for q, rows in cands:
            if np.linalg.norm(np.nanmean(q, 0) - self.centre) >= self.radius_mm:
                continue
            c2 = self._centre2d(q)
            if any(np.linalg.norm(c2[r] - self.home[r]) >= self.radius_px for r in rows):
                continue
            if best is None or np.isfinite(q[:, 0]).sum() > np.isfinite(best[:, 0]).sum():
                best = q
        p = self.filter.step(best if best is not None else np.full((21, 3), np.nan))
        return p if np.isfinite(p[:, 0]).any() else None

    def _learn(self):
        import cv2
        cen = np.array([np.nanmean(q, 0) for q in self.warm], np.float32)
        cv2.setRNGSeed(0)
        _, lab, cc = cv2.kmeans(cen, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1), 5,
                                cv2.KMEANS_PP_CENTERS)
        r = self.order_row if self.order_row is not None else self.nrow - 1
        order = np.argsort(self.cg.project(cc.astype(float))[r][:, 0])
        k = order[self.hand]
        self.centre = cc[k].astype(float)
        mine = [q for q, l in zip(self.warm, lab.ravel()) if l == k and np.linalg.norm(np.nanmean(q, 0) - self.centre) < self.radius_mm]
        c2 = [self._centre2d(q) for q in mine]
        self.home = {rr: np.nanmedian([c[rr] for c in c2], 0) for rr in range(self.nrow)}
        logger.info(f"held hand place learned from {len(self.warm)} warm-up reconstructions: centre {self.centre.round()} mm; "
                    f"hands numbered left->right in calibration row {r}, using hand {self.hand}")
        self.warm = []

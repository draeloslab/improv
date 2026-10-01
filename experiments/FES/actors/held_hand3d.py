"""3D of ONE hand that stays in a known place (e.g. the stimulated hand an experimenter holds), from a
two-hand DLC model, frame by frame. `hand_association: region` in ProcessorBatch3D.

Why not hand_assoc3d.py: DLC gives a keypoint-level likelihood and its right/left labels are unreliable
across cameras, and the held hand is partly hidden, so whole-hand matching fails. Per step (after warm-up):
  1. in each camera, of its two DLC hands take the one whose image centre (median of keypoints with
     likelihood >= `likelihood`) is nearest the target hand's usual image position, within `radius_px`
     (with only two cameras a wrong match can reproject perfectly, e.g. the monkey's chin, so the image
     position matters); cameras with neither hand there sit the frame out;
  2. triangulate that one assignment (keypoints must reproject within `max_px` in every camera that sees
     them) and keep it if its 3D centre is within `radius_mm` of the hand's place. One triangulation per
     step, whatever the number of cameras (it used to try every combination: 2^cameras);
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
    """Per-keypoint One-Euro filter for (n, 3) points; a keypoint missing for more than max_gap frames is dropped."""

    def __init__(self, n=21, min_cutoff=1.0, beta=0.05, d_cutoff=1.0, max_gap=5, fps=30.0):
        self.mc, self.beta, self.dc, self.max_gap, self.fps = min_cutoff, beta, d_cutoff, max_gap, fps
        self.x = np.full((n, 3), np.nan)
        self.dx = np.zeros((n, 3))
        self.gap = np.zeros(n, int)

    def step(self, z):
        """(n, 3) measurement (NaN = missing) -> filtered points (NaN where this frame had none)."""
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
    """3D of the one hand that stays in a learned place (config `held_hand`, hand_association: region)."""

    def __init__(self, cgroup, likelihood=0.3, max_px=15.0, warmup_s=10.0, hand=1, order_row=None,
                 radius_mm=80.0, radius_px=40.0, smooth=None, fps=30.0):
        """Config held_hand: likelihood / max_px gate keypoints; warmup_s learns the hand's place; hand picks which of
        two (numbered left -> right in calibration row order_row); radius_mm / radius_px gate it afterwards."""
        self.cg = cgroup
        self.nrow = len(cgroup.cameras)
        self.lik, self.max_px, self.warmup_s, self.hand = likelihood, max_px, warmup_s, hand
        self.order_row = order_row
        self.radius_mm, self.radius_px = radius_mm, radius_px
        self.filter = OneEuroKeypoints(fps=fps, **(smooth or {}))
        self.t0 = None
        self.warm = []
        self.centre = self.home = None

    def _solve(self, views, pick):
        """Triangulate the hands picked per camera ({calib_row: 0 or 21}). Keypoints need likelihood >= lik in
        >= 2 cameras and must reproject within max_px in every camera that sees them. (21, 3) or None."""
        pts = np.full((self.nrow, 21, 2), np.nan)
        for r, off in pick.items():
            h = views[r][off:off + 21]
            ok = h[:, 2] >= self.lik
            pts[r][ok] = h[ok, :2]
        seen = np.isfinite(pts[..., 0]).sum(0) >= 2
        if seen.sum() < 3:
            return None
        pts[:, ~seen] = np.nan
        q = self.cg.triangulate(pts, progress=False)
        e = np.linalg.norm(self.cg.project(q) - pts, axis=-1)
        ok = (np.nanmax(np.where(np.isfinite(e), e, 0), axis=0) <= self.max_px) & seen
        return np.where(ok[:, None], q, np.nan) if ok.sum() >= 3 else None

    def _candidates(self, views):
        """Warm-up only: every combination of one DLC hand per camera, over the (at most) 3 cameras with the
        most confident keypoints -- 2^3 triangulations instead of 2^cameras. Yields (3D (21, 3), rows)."""
        conf = {r: (v[:, 2] >= self.lik).sum() for r, v in views.items()}
        rows = sorted(views, key=lambda r: -conf[r])[:3]
        for pick in itertools.product((0, 21), repeat=len(rows)):
            q = self._solve(views, dict(zip(rows, pick)))
            if q is not None:
                yield q, rows

    def _hand_centre(self, h):
        """Median image position of a (21, 3) DLC hand's confident keypoints, or None."""
        ok = h[:, 2] >= self.lik
        return np.median(h[ok, :2], 0) if ok.sum() >= 3 else None

    def _tracked(self, views):
        """After warm-up: in each camera, the DLC hand whose image centre is nearest the held hand's usual
        position (within radius_px), then ONE triangulation; the 3D centre must be within radius_mm of its place."""
        pick = {}
        for r, v in views.items():
            best = None
            for off in (0, 21):
                c = self._hand_centre(v[off:off + 21])
                if c is None:
                    continue
                d = np.linalg.norm(c - self.home[r])
                if d < self.radius_px and (best is None or d < best[0]):
                    best = (d, off)
            if best is not None:
                pick[r] = best[1]
        if len(pick) < 2:
            return None
        q = self._solve(views, pick)
        if q is None or np.linalg.norm(np.nanmean(q, 0) - self.centre) >= self.radius_mm:
            return None
        return q

    def _centre2d(self, q):
        """A 3D hand's median projected position in every camera: {calibration row: (x, y)}."""
        return {r: np.nanmedian(self.cg.project(q)[r], 0) for r in range(self.nrow)}

    def step(self, t, views):
        """t: capture time (s); views: {calib_row: (42, 3)}. Returns (21, 3) mm or None."""
        if self.t0 is None:
            self.t0 = t
        if self.centre is None:
            if len(views) >= 2:
                self.warm += [q for q, _ in self._candidates(views)]
            if t - self.t0 >= self.warmup_s and len(self.warm) >= 20:
                self._learn()
            return None
        best = self._tracked(views) if len(views) >= 2 else None
        p = self.filter.step(best if best is not None else np.full((21, 3), np.nan))
        return p if np.isfinite(p[:, 0]).any() else None

    def _learn(self):
        """End of warm-up: cluster the warm-up hands into two places and keep the `hand`-th from the left."""
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

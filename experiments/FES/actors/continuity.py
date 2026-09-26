"""Continuous (causal) 3D output: `continuity` in config.yaml, applied by ProcessorBatch3D to every
step's points before joint angles, so the GUI, the angles and the UDP sender never see gaps or steps.

Per keypoint: hold the last value through a gap; when it comes back (or jumps more than jump_mm,
e.g. another hand took over) glide to the new measurement over ~glide_frames (exponential,
glide_alpha per frame) instead of stepping. Continuously measured keypoints pass straight through.
NaN only before a keypoint is first seen. Same rule as analysis/fes_figures.continuous().
"""
import numpy as np


class ContinuousPose:
    def __init__(self, n_keypoints, glide_frames=20, glide_alpha=0.15, jump_mm=40.0):
        self.gf, self.ga, self.jump = glide_frames, glide_alpha, jump_mm
        self.prev = np.full((n_keypoints, 3), np.nan)
        self.glide = np.zeros(n_keypoints, int)

    def step(self, z):
        prev = self.prev
        have = np.isfinite(z[:, 0])
        known = np.isfinite(prev[:, 0])
        first = have & ~known
        prev[first] = z[first]
        jump = have & known & (np.linalg.norm(np.nan_to_num(z - prev), axis=1) > self.jump)
        self.glide[~have & known] = self.gf
        self.glide[jump] = self.gf
        upd = have & known
        a = np.where(self.glide[upd] > 0, self.ga, 1.0)[:, None]
        prev[upd] += a * (z[upd] - prev[upd])
        self.glide[upd] = np.maximum(self.glide[upd] - 1, 0)
        return prev.copy()

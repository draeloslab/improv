"""
3D Kalman smoothing and short-gap prediction of keypoints (config `smoothing_3d: {method: kalman}`).

Replaces the One-Euro filter + hold-and-glide chain on the final 3D points. Per keypoint and axis, a constant-velocity
Kalman filter (state: position, velocity) driven by the real time between steps, so it does not depend on the configured
frame rate:

  measured      predict + update: smooths with little lag (a constant-velocity model has no lag at constant speed);
  missing       predict only, with the velocity decaying (time constant coast_tau_s) so a lost keypoint coasts to a stop
                rather than flying off -- the same idea as DLC-Live's Kalman filter (actors/kalmanfilter.py) in 2D;
  missing for longer than max_predict_s
                stop predicting and hold the last position (hold_s: null = until found again, else NaN after hold_s);
  found again after a hold, or more than reset_mm from the prediction
                snap to the measurement (velocity 0) instead of gliding: the output shows where the hand is.

    kf = KalmanPose3D(n_keypoints=42, measurement_sd_mm=3, accel_sd_mm_s2=1000)
    smoothed = kf.step(points_3d, t)        # (K, 3) mm, NaN = missing; t in seconds
"""
import numpy as np


class KalmanPose3D:
    """Constant-velocity Kalman filter for every keypoint of a (K, 3) stream; NaN rows are missing measurements."""

    def __init__(self, n_keypoints, measurement_sd_mm=3.0, accel_sd_mm_s2=1000.0, max_predict_s=0.2,
                 coast_tau_s=0.1, reset_mm=60.0, hold_s=None, max_dt_s=0.5):
        """measurement_sd_mm: triangulation noise (higher = smoother, laggier). accel_sd_mm_s2: how hard the hand can
        accelerate (higher = follows faster, smooths less). max_predict_s: how long a missing keypoint is extrapolated.
        coast_tau_s: velocity decay while extrapolating. reset_mm: innovation above which the filter snaps to the
        measurement. hold_s: how long a lost keypoint is held after prediction stops (None = until found).
        max_dt_s: longer steps (pauses) are treated as a restart."""
        self.r = float(measurement_sd_mm) ** 2
        self.q = float(accel_sd_mm_s2) ** 2
        self.max_predict_s, self.coast_tau_s = float(max_predict_s), float(coast_tau_s)
        self.reset_mm, self.hold_s, self.max_dt_s = float(reset_mm), hold_s, float(max_dt_s)
        self.p = np.full((n_keypoints, 3), np.nan)       # position
        self.v = np.zeros((n_keypoints, 3))              # velocity
        self.P = np.zeros((n_keypoints, 2, 2))           # covariance of (position, velocity), shared by the 3 axes
        self.missing_s = np.zeros(n_keypoints)           # time since each keypoint was last measured
        self.t = None

    def _reset(self, k, z):
        """Start keypoints k at measurement z with zero velocity."""
        self.p[k], self.v[k] = z, 0.0
        self.P[k] = np.diag([self.r, 1e6])
        self.missing_s[k] = 0.0

    def step(self, z, t):
        """One step. z: (K, 3) measured points (NaN = missing), t: time (s). Returns the filtered (K, 3) points."""
        z = np.asarray(z, float)
        dt = 0.0 if self.t is None else float(t - self.t)
        self.t = t
        have = np.isfinite(z).all(1)
        known = np.isfinite(self.p[:, 0])
        if dt <= 0 or dt > self.max_dt_s:                 # first step or a pause: restart from the measurements
            self._reset(have, z[have])
            self.p[~have] = np.nan
            return self.p.copy()

        # predict (only keypoints still allowed to move: measured ones, and missing ones within max_predict_s)
        self.missing_s[~have] += dt
        moving = known & (have | (self.missing_s <= self.max_predict_s))
        coast = moving & ~have
        self.v[coast] *= np.exp(-dt / self.coast_tau_s)
        self.p[moving] += self.v[moving] * dt
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = self.q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        self.P[moving] = F @ self.P[moving] @ F.T + Q
        stopped = known & ~have & (self.missing_s > self.max_predict_s)
        self.v[stopped] = 0.0
        if self.hold_s is not None:
            self.p[known & ~have & (self.missing_s > self.max_predict_s + self.hold_s)] = np.nan

        # update
        upd = have & np.isfinite(self.p[:, 0])
        new = have & ~upd                                 # first sighting, or dropped after hold_s
        returning = upd & (self.missing_s > self.max_predict_s)
        jump = upd & (np.linalg.norm(np.nan_to_num(z - self.p), axis=1) > self.reset_mm)
        snap = new | returning | jump
        upd &= ~snap
        if upd.any():
            S = self.P[upd, 0, 0] + self.r                # innovation variance (position measured)
            K = self.P[upd, :, 0] / S[:, None]            # (n, 2) gains for position and velocity
            y = z[upd] - self.p[upd]
            self.p[upd] += K[:, :1] * y
            self.v[upd] += K[:, 1:] * y
            I_KH = np.eye(2)[None] - K[:, :, None] * np.array([1.0, 0.0])[None, None, :]
            self.P[upd] = I_KH @ self.P[upd]
        if snap.any():
            self._reset(snap, z[snap])
        self.missing_s[have] = 0.0
        return self.p.copy()

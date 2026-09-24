"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import time
import numpy as np
# from dlclive.processor import Processor


class KalmanFilterPredictor():
    def __init__(
        self,
        adapt=True,
        forward=0.002,
        fps=30,
        nderiv=2,
        priors=[10, 10],
        initial_var=5,
        process_var=5,
        dlc_var=20,
        lik_thresh=0.5,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.adapt = adapt
        self.forward = forward
        self.dt = 1.0 / fps
        self.nderiv = nderiv
        self.priors = np.hstack(([1e5], priors))
        self.initial_var = initial_var
        self.process_var = process_var
        self.dlc_var = dlc_var
        self.lik_thresh = lik_thresh
        self.is_initialized = False
        self.last_pose_time = 0

    def _get_forward_model(self, dt):

        F = np.zeros((self.n_states, self.n_states))
        for d in range(self.nderiv + 1):
            for i in range(self.n_states - (d * self.bp * 2)):
                F[i, i + (2 * self.bp * d)] = (dt ** d) / max(1, d)

        return F

    def _init_kf(self, pose):

        # get number of body parts
        self.bp = pose.shape[0]
        self.n_states = self.bp * 2 * (self.nderiv + 1)

        # initialize state matrix, set position to first pose
        self.X = np.zeros((self.n_states, 1))
        self.X[: (self.bp * 2)] = pose[:, :2].reshape(self.bp * 2, 1)

        # initialize covariance matrix, measurement noise and process noise
        self.P = np.eye(self.n_states) * self.initial_var
        self.R = np.eye(self.n_states) * self.dlc_var
        self.Q = np.eye(self.n_states) * self.process_var

        self.H = np.eye(self.n_states)
        self.K = np.zeros((self.n_states, self.n_states))
        self.I = np.eye(self.n_states)

        # initialize priors for forward prediction step only
        B = np.repeat(self.priors, self.bp * 2)
        self.B = B.reshape(B.size, 1)

        self.is_initialized = True

    def _predict(self):

        F = self._get_forward_model(time.time() - self.last_pose_time)

        Pd = np.diag(self.P).reshape(self.P.shape[0], 1)
        X = (1 / ((1 / Pd) + (1 / self.B))) * (self.X / Pd)

        self.Xp = np.dot(F, X)
        self.Pp = np.dot(np.dot(F, self.P), F.T) + self.Q

    def _get_residuals(self, pose):

        z = np.zeros((self.n_states, 1))
        z[: (self.bp * 2)] = pose[: self.bp, :2].reshape(self.bp * 2, 1)
        for i in range(self.bp * 2, self.n_states):
            z[i] = (z[i - (self.bp * 2)] - self.X[i - (self.bp * 2)]) / self.dt
        self.y = z - np.dot(self.H, self.Xp)

    def _update(self, liks):

        S = np.dot(self.H, np.dot(self.Pp, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pp, self.H.T), np.linalg.inv(S))
        self.X = self.Xp + np.dot(K, self.y)
        self.X[liks < self.lik_thresh] = self.Xp[liks < self.lik_thresh]
        self.P = np.dot(self.I - np.dot(K, self.H), self.Pp)

    def _get_future_pose(self, dt):

        Ff = self._get_forward_model(time.time() - self.last_pose_time)
        Xf = np.dot(Ff, self.X)
        future_pose = Xf[: (self.bp * 2)].reshape(self.bp, 2)

        return future_pose

    def _get_state_likelihood(self, pose):

        liks = pose[:, 2]
        liks_xy = np.repeat(liks, 2)
        liks_xy_deriv = np.tile(liks_xy, self.nderiv + 1)
        liks_state = liks_xy_deriv.reshape(liks_xy_deriv.shape[0], 1)
        return liks_state

    def process(self, pose, **kwargs):

        if not self.is_initialized:

            self._init_kf(pose)
            self.last_pose_time = time.time()
            return pose

        else:

            self._predict()
            self._get_residuals(pose)
            liks = self._get_state_likelihood(pose)
            self._update(liks)

            forward_time = (
                (time.time() - kwargs["frame_time"] + self.forward)
                if self.adapt
                else self.forward
            )

            future_pose = self._get_future_pose(forward_time)
            future_pose = np.hstack((future_pose, pose[:, 2].reshape(self.bp, 1)))

            self.last_pose_time = time.time()
            return future_pose


class KalmanHandSmoother:
    """Continuous 2D keypoints from a stream that drops out, built on the DLC
    KalmanFilterPredictor above.

    Per frame, each keypoint is either MEASURED (finite and likelihood >=
    lik_thresh) or ESTIMATED (the filter's own prediction, which coasts and, via
    the DLC priors, decays its velocity so a lost hand settles rather than
    flying off). Estimated keypoints come back with likelihood `coast_lik`, so
    the GUI can colour them and the downstream gates keep them. After
    `max_coast` estimated frames in a row a keypoint is dropped (NaN) and the
    next measurement re-seeds it.

    Unlike KalmanFilterPredictor.process() this does NOT extrapolate forward in
    time (that is latency compensation, not smoothing), and a keypoint that comes
    back after being lost snaps to the measurement instead of being dragged
    there by the gain.
    """

    def __init__(self, fps=30, lik_thresh=0.6, coast_lik=0.5, max_coast=15,
                 priors=(1, 1), initial_var=10, process_var=1, dlc_var=10, nderiv=2):
        self.kf = KalmanFilterPredictor(
            adapt=False, forward=0.0, fps=fps, nderiv=nderiv, priors=list(priors),
            initial_var=initial_var, process_var=process_var, dlc_var=dlc_var,
            lik_thresh=lik_thresh)
        self.lik_thresh = lik_thresh
        self.coast_lik = coast_lik
        self.max_coast = max_coast
        self.age = None        # frames since each keypoint was last measured

    def process(self, pose):
        """pose: (K, 3) x, y, likelihood, NaN where nothing was found."""
        kf = self.kf
        pose = np.asarray(pose, dtype=float)
        K = len(pose)
        measured = (np.isfinite(pose[:, :2]).all(axis=1)
                    & (np.nan_to_num(pose[:, 2], nan=0.0) >= self.lik_thresh))

        if not kf.is_initialized:
            seed = pose.copy()
            seed[~np.isfinite(seed)] = 0.0
            kf._init_kf(seed)
            kf.last_pose_time = time.time()
            self.age = np.full(K, self.max_coast + 1)   # nothing seen yet

        reseed = measured & (self.age > 0)              # new or returning keypoint
        if reseed.any():
            for k in np.flatnonzero(reseed):
                kf.X[2 * k:2 * k + 2, 0] = pose[k, :2]
                for d in range(1, kf.nderiv + 1):
                    kf.X[2 * K * d + 2 * k:2 * K * d + 2 * k + 2, 0] = 0.0

        # Unmeasured keypoints feed the filter its own last estimate at lik 0, so
        # _update() keeps the prediction for them.
        feed = pose.copy()
        est = kf.X[:2 * K, 0].reshape(K, 2)
        feed[~measured, :2] = est[~measured]
        feed[~measured, 2] = 0.0

        kf._predict()
        kf._get_residuals(feed)
        kf._update(kf._get_state_likelihood(feed))
        kf.last_pose_time = time.time()

        self.age = np.where(measured, 0, np.minimum(self.age + 1, self.max_coast + 1))
        out = np.full((K, 3), np.nan)
        state = kf.X[:2 * K, 0].reshape(K, 2)
        alive = self.age <= self.max_coast
        out[alive, :2] = state[alive]
        out[alive, 2] = np.where(measured[alive], pose[alive, 2], self.coast_lik)
        return out

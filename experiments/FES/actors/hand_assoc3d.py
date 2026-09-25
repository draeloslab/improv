"""Geometric cross-camera hand association, tracking and 3D stabilisation.

Replaces "match hands across cameras by MediaPipe's handedness label", which on
the rig pairs the monkey's hand in one camera with an experimenter's glove or the
other hand in another (09-25 130900: ~95 px median reprojection error, bone
lengths varying ~60%). Here a hand only exists in 3D if its detections in
different cameras actually agree geometrically.

Per frame (HandTracker3D.step):
  1. every pair of detections from different cameras is triangulated (one
     batched DLT); pairs whose mean reprojection error is < tol_px and whose
     wrist->middle-MCP length is plausible seed a 3D hand;
  2. greedily (best pair first) other cameras' detections join a hand if they
     reproject within tol_px; each detection is used once;
  3. hands are matched to tracks by palm-centre distance (Hungarian, gate_mm);
     a track's side (right/left) is the sign of its accumulated handedness votes
     (score-weighted, across cameras and frames), so one camera's wrong label
     no longer flips a hand;
  4. each side's output goes through a One-Euro filter per keypoint (MediaPipe's
     own landmark smoother); when a side is lost the last hand is held whole for
     hold_frames, and the filter resets when the palm jumps > reset_mm.

Tuned offline with scripts/tune3d (evaluate.py uses this module).
"""
import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment

PALM = [0, 5, 9, 13, 17]          # wrist + the four MCPs (MediaPipe indices)


class OneEuroHand:
    """One-Euro filter over a (21, 3) hand, with whole-hand hold and reset."""

    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0, fps=30.0,
                 hold_frames=5, reset_mm=60.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.fps, self.hold_frames, self.reset_mm = fps, hold_frames, reset_mm
        self.x = self.dx = None
        self.held = 0

    def _alpha(self, fc):
        return 1.0 / (1.0 + self.fps / (2 * np.pi * fc))

    def step(self, z):
        """z: (21, 3) or None. Returns the filtered hand or None."""
        if z is None or np.isfinite(z[:, 0]).sum() < 15:
            if self.x is not None and self.held < self.hold_frames:
                self.held += 1
                return self.x.copy()
            self.x = None
            return None
        self.held = 0
        if self.x is not None:
            jump = np.linalg.norm(np.nanmean(z[PALM], 0) - np.nanmean(self.x[PALM], 0))
            if not np.isfinite(jump) or jump > self.reset_mm:
                self.x = None
        if self.x is None:
            self.x = z.copy()
            self.dx = np.zeros_like(z)
            return self.x.copy()
        zz = np.where(np.isfinite(z), z, self.x)        # a missing keypoint keeps its last value
        self.dx = self.dx + self._alpha(self.d_cutoff) * ((zz - self.x) * self.fps - self.dx)
        self.x = self.x + self._alpha(self.min_cutoff + self.beta * np.abs(self.dx)) * (zz - self.x)
        return self.x.copy()


class HandTracker3D:
    """cgroup: aniposelib CameraGroup. Detections are given per frame as a list of
    (calib_row, xy (21, 2) in calibration pixels, label 0=right/1=left, score)."""

    def __init__(self, cgroup, tol_px=15.0, size_mm=(20.0, 200.0), gate_mm=100.0,
                 max_miss=10, smooth=None, fps=30.0):
        self.cg = cgroup
        self.nrows = len(cgroup.cameras)
        self.tol, self.size_mm, self.gate, self.max_miss = tol_px, size_mm, gate_mm, max_miss
        self.tracks = []
        self.next_id = 0
        smooth = {} if smooth is None else smooth
        self.smooth_on = smooth is not False
        self.filters = [OneEuroHand(fps=fps, **(smooth or {})) for _ in range(2)]
        self.last_errors = []

    # ------------------------------------------------------------ geometry
    def _triangulate(self, views):
        p = np.full((self.nrows, 21, 2), np.nan)
        for r, xy in views:
            p[r] = xy
        return self.cg.triangulate(p, progress=False)

    def _view_error(self, p3, row, xy):
        proj = self.cg.project(p3)[row]
        return float(np.nanmean(np.linalg.norm(proj - xy, axis=1)))

    def associate(self, dets):
        """Returns a list of hands: dict(p3, members=[det indices], votes, err)."""
        pairs = [(a, b) for a, b in itertools.combinations(range(len(dets)), 2)
                 if dets[a][0] != dets[b][0]]
        if not pairs:
            return []
        P = len(pairs)
        arr = np.full((self.nrows, P, 21, 2), np.nan)
        for j, (a, b) in enumerate(pairs):
            arr[dets[a][0], j] = dets[a][1]
            arr[dets[b][0], j] = dets[b][1]
        p3 = self.cg.triangulate(arr.reshape(self.nrows, -1, 2), progress=False).reshape(P, 21, 3)
        proj = self.cg.project(p3.reshape(-1, 3)).reshape(self.nrows, P, 21, 2)
        cand = []
        for j, (a, b) in enumerate(pairs):
            e = np.mean([np.nanmean(np.linalg.norm(proj[dets[x][0], j] - dets[x][1], axis=1))
                         for x in (a, b)])
            size = np.linalg.norm(p3[j, 9] - p3[j, 0])       # wrist -> middle MCP
            if e < self.tol and self.size_mm[0] <= size <= self.size_mm[1]:
                cand.append((e, a, b))
        cand.sort()
        used, hands = set(), []
        for _, a, b in cand:
            if a in used or b in used:
                continue
            members = [a, b]
            rows = {dets[a][0], dets[b][0]}
            views = [(dets[a][0], dets[a][1]), (dets[b][0], dets[b][1])]
            p = self._triangulate(views)
            for x in range(len(dets)):
                if x in used or x in members or dets[x][0] in rows:
                    continue
                if self._view_error(p, dets[x][0], dets[x][1]) < self.tol:
                    members.append(x); rows.add(dets[x][0])
                    views.append((dets[x][0], dets[x][1]))
            if len(members) > 2:
                p = self._triangulate(views)
            used.update(members)
            votes = sum((1.0 if dets[x][2] == 1 else -1.0) * dets[x][3] for x in members)   # + = left
            err = float(np.mean([self._view_error(p, r, xy) for r, xy in views]))
            hands.append(dict(p3=p, members=members, votes=votes, err=err))
        return hands

    # ------------------------------------------------------------ tracking
    def step(self, dets):
        """Returns (points_3d (42, 3) right block then left, side_of_det list:
        0/1 for detections that ended up in an output hand, else None)."""
        hands = self.associate(dets) if len(dets) >= 2 else []
        self.last_errors = [h['err'] for h in hands]
        pairs = []
        if self.tracks and hands:
            C = np.array([[np.nanmean(np.linalg.norm(t['p3'][PALM] - h['p3'][PALM], axis=1))
                           for h in hands] for t in self.tracks])
            C = np.nan_to_num(C, nan=1e9)
            ti, hi = linear_sum_assignment(C)
            pairs = [(a, b) for a, b in zip(ti, hi) if C[a, b] < self.gate]
        matched_t = {a for a, _ in pairs}
        matched_h = {b for _, b in pairs}
        for a, b in pairs:
            t = self.tracks[a]
            t.update(p3=hands[b]['p3'], miss=0, votes=t['votes'] + hands[b]['votes'],
                     age=t['age'] + 1, hand=hands[b])
        for a, t in enumerate(self.tracks):
            if a not in matched_t:
                t['miss'] += 1; t['hand'] = None
        self.tracks = [t for t in self.tracks if t['miss'] <= self.max_miss]
        for b, h in enumerate(hands):
            if b not in matched_h:
                self.tracks.append(dict(p3=h['p3'], miss=0, votes=h['votes'], age=1,
                                        id=self.next_id, hand=h))
                self.next_id += 1

        out = np.full((42, 3), np.nan)
        side_of_det = [None] * len(dets)
        live = [t for t in self.tracks if t['miss'] == 0]
        for side in (0, 1):
            cands = [t for t in live if (t['votes'] > 0) == bool(side)]
            best = max(cands, key=lambda t: t['age']) if cands else None
            raw = best['p3'] if best is not None else None
            if best is not None:
                for x in best['hand']['members']:
                    side_of_det[x] = side
            p = self.filters[side].step(raw) if self.smooth_on else raw
            if p is not None:
                out[side * 21:side * 21 + 21] = p
        return out, side_of_det


def glove_mask(frame_hsv, xy, teal=(85, 112, 60), white=None):
    """True if the hand's keypoints sit on an experimenter's glove (OpenCV HSV).
    Measured 09-25: teal nitrile glove H~98 S~100; monkey hand H~8 S~48 V~47."""
    h, w = frame_hsv.shape[:2]
    samples = []
    for x, y in np.asarray(xy, dtype=float):
        if np.isfinite(x) and 2 <= x < w - 2 and 2 <= y < h - 2:
            xi, yi = int(x), int(y)
            samples.append(frame_hsv[yi - 2:yi + 3, xi - 2:xi + 3].reshape(-1, 3))
    if not samples:
        return False
    hh, ss, vv = np.median(np.concatenate(samples), axis=0)
    if teal is not None and teal[0] <= hh <= teal[1] and ss >= teal[2]:
        return True
    if white is not None and ss <= white[0] and vv >= white[1]:
        return True
    return False

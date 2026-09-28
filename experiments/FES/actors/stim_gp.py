"""Gaussian-process model of stimulation -> joint-angle change, for closed-loop Bayesian optimisation.

The same model as analysis/stim_gp/stim_gp_walkthrough.ipynb (section 2), packaged for live use by
actors/bayes_opt.py: one GP per joint, input = which electrodes are on + pulse width, frequency, current.

    k = k_z(z, z') x [ c^2 + sum_e a_e^2 e_e e'_e + s^2 exp(-|e - e'|^2 / 2 l^2) ] + noise^2 (same condition)

e: 1 per active electrode; z = log2(pulse width / 100), log2(frequency / 30), current / 5000 - 1.
Hyper-parameters are fitted by marginal likelihood (L-BFGS-B, a few starts); between refits new
observations only update the Cholesky factor, which is cheap.
"""
import itertools
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ELECTRODES = [2, 4, 6, 8, 10, 12, 14, 16]
NE = len(ELECTRODES)
NP = NE + 7                               # c, a (8), s, l, lambda (pulse width, frequency, current), noise
# lambda >= 0.5 (pulse width), >= 1 (frequency, current); noise >= 10% of the joint's sd (see the notebook)
BOUNDS = [(-5, 3)] * (NE + 3) + [(np.log(0.5), 3)] + [(np.log(1.0), 3)] * 2 + [(np.log(0.1), 1)]


def features(electrodes, pulse_width, frequency, amplitude):
    """electrodes: list of electrode lists; the rest array-like of the same length -> (E (n, 8), Z (n, 3))."""
    E = np.array([[e in els for e in ELECTRODES] for els in electrodes], float).reshape(-1, NE)
    Z = np.c_[np.log2(np.asarray(pulse_width, float) / 100), np.log2(np.asarray(frequency, float) / 30),
              np.asarray(amplitude, float) / 5000 - 1]
    return E, Z


def unpack(th):
    return dict(c=np.exp(th[0]), a=np.exp(th[1:1 + NE]), s=np.exp(th[1 + NE]), l=np.exp(th[2 + NE]),
                lam=np.exp(th[3 + NE:6 + NE]), noise=np.exp(th[6 + NE]))


def kernel(p, E1, Z1, E2, Z2):
    kz = np.exp(-0.5 * (((Z1[:, None, :] - Z2[None, :, :]) / p['lam']) ** 2).sum(-1))
    add = (E1 * p['a'] ** 2) @ E2.T
    hamming = E1.sum(1)[:, None] + E2.sum(1)[None, :] - 2 * E1 @ E2.T      # binary vectors: squared distance
    return kz * (p['c'] ** 2 + add + p['s'] ** 2 * np.exp(-0.5 * hamming / p['l'] ** 2))


class JointGP:
    """One joint's GP. Data are kept raw (deg); the fit standardises internally."""

    def __init__(self):
        self.E = np.zeros((0, NE)); self.Z = np.zeros((0, 3)); self.y = np.zeros(0)
        self.th = np.r_[0.0, np.full(NE, -0.5), -0.5, 0.5, 0.0, 0.0, 0.0, -0.7]
        self.mu, self.sd = 0.0, 10.0

    @property
    def p(self):
        return unpack(self.th)

    def add(self, E, Z, y):
        m = np.isfinite(y)
        self.E = np.vstack([self.E, E[m]]); self.Z = np.vstack([self.Z, Z[m]]); self.y = np.r_[self.y, y[m]]
        self._factor()

    def fit(self, starts=4, seed=0):
        """Marginal-likelihood fit of the hyper-parameters (standardised y)."""
        if len(self.y) < 5:
            self._factor(); return self
        self.mu, self.sd = float(self.y.mean()), float(self.y.std() or 1.0)
        ys = (self.y - self.mu) / self.sd
        rng = np.random.default_rng(seed); best = None
        for k in range(starts):
            th0 = self.th + (rng.normal(0, 0.7, NP) if k else 0)
            r = minimize(self._nll, np.clip(th0, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS]), args=(ys,),
                         method='L-BFGS-B', bounds=BOUNDS)
            if best is None or r.fun < best.fun:
                best = r
        self.th = best.x
        self._factor()
        return self

    def _nll(self, th, ys):
        p = unpack(th)
        K = kernel(p, self.E, self.Z, self.E, self.Z) + (p['noise'] ** 2 + 1e-6) * np.eye(len(ys))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        a = np.linalg.solve(L.T, np.linalg.solve(L, ys))
        return 0.5 * ys @ a + np.log(np.diag(L)).sum()

    def _factor(self):
        p = self.p
        if len(self.y) == 0:
            self.L = None; return
        K = kernel(p, self.E, self.Z, self.E, self.Z) + (p['noise'] ** 2 + 1e-6) * np.eye(len(self.y))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, (self.y - self.mu) / self.sd))

    def predict(self, E, Z):
        """Mean and sd (deg) of the joint change at each input row (the prior's, if there are no data yet)."""
        p = self.p
        kxx = p['c'] ** 2 + (E * p['a'] ** 2).sum(1) + p['s'] ** 2
        if self.L is None:
            return np.full(len(E), self.mu), self.sd * np.sqrt(kxx)
        Ks = kernel(p, E, Z, self.E, self.Z)
        v = np.linalg.solve(self.L, Ks.T)
        return self.mu + self.sd * Ks @ self.alpha, self.sd * np.sqrt(np.maximum(kxx - (v ** 2).sum(0), 0))

    @property
    def noise_deg(self):
        return float(self.p['noise'] * self.sd)


class StimModel:
    """GPs for several joints over one candidate grid."""

    def __init__(self, joints, electrodes_allowed=ELECTRODES, pulse_widths=(80, 100, 120, 160, 200),
                 frequencies=(20, 30, 50, 100), currents=(1000, 2000, 3000, 4000, 5000), max_electrodes=8):
        self.joints = list(joints)
        self.gps = {j: JointGP() for j in self.joints}
        combos = [list(c) for n in range(1, max_electrodes + 1) for c in itertools.combinations(sorted(electrodes_allowed), n)]
        self.grid = [(c, pw, fq, cu) for c in combos for pw in pulse_widths for fq in frequencies for cu in currents]
        self.Ec, self.Zc = features([g[0] for g in self.grid], [g[1] for g in self.grid],
                                    [g[2] for g in self.grid], [g[3] for g in self.grid])

    def add(self, electrodes, pulse_width, frequency, amplitude, changes):
        """changes: {joint: deg change} for ONE stimulation (NaN / missing joints are skipped)."""
        E, Z = features([electrodes], [pulse_width], [frequency], [amplitude])
        for j in self.joints:
            self.gps[j].add(E, Z, np.array([changes.get(j, np.nan)], float))

    def add_table(self, rows):
        """rows: iterable of dicts with electrodes (list), pulse_width, frequency, amplitude and joint columns."""
        for r in rows:
            self.add(r['electrodes'], r['pulse_width'], r['frequency'], r['amplitude'], r)

    def fit(self, starts=4):
        for g in self.gps.values():
            g.fit(starts=starts)

    def predict_grid(self):
        """(mean (n_cand, n_joints), sd (n_cand, n_joints)) of the joint changes for every candidate."""
        out = [self.gps[j].predict(self.Ec, self.Zc) for j in self.joints]
        return np.stack([o[0] for o in out], 1), np.stack([o[1] for o in out], 1)

    def n_obs(self):
        return {j: len(g.y) for j, g in self.gps.items()}


def load_prior_runs(folders, joint_prefix=''):
    """Rows from analysed runs' stim_joint_response.csv (analysis/stim_1153/walkthrough_out*), joint names
    prefixed (e.g. 'left_') to match the live processor's joint names."""
    import pandas as pd
    rows = []
    for f in folders:
        path = Path(f) / 'stim_joint_response.csv'
        if not path.exists():
            continue
        d = pd.read_csv(path, dtype={'electrodes': str})       # else '2' (single electrodes) parses as 2.0
        if 'frequency' not in d:
            continue
        for _, r in d.iterrows():
            row = dict(electrodes=[int(float(e)) for e in str(r['electrodes']).split('-')], pulse_width=float(r['pulse_width']),
                       frequency=float(r['frequency']), amplitude=float(r['amplitude']))
            for c in d.columns:
                if c.isupper():
                    row[joint_prefix + c] = float(r[c])
            rows.append(row)
    return rows


def window_change(t, A, t_on, t_off, settle=0.5, baseline=1.0, min_cover=0.5):
    """Joint change for one stimulation, the analysis notebook's definition: mean over [t_on + settle, t_off)
    minus mean over [t_on - baseline, t_on). t: (n,) frame times, A: (n, n_joints). Returns (change, base, cover)
    per joint; NaN where fewer than min_cover of the window's frames measured that joint."""
    t = np.asarray(t); A = np.asarray(A, float)
    pre, dur = (t >= t_on - baseline) & (t < t_on), (t >= t_on + settle) & (t < t_off)
    out = np.full(A.shape[1], np.nan); base = np.full(A.shape[1], np.nan); cover = np.zeros(A.shape[1])
    if pre.sum() == 0 or dur.sum() == 0:
        return out, base, cover
    fp, fd = np.isfinite(A[pre]).mean(0), np.isfinite(A[dur]).mean(0)
    cover = np.minimum(fp, fd)
    ok = cover >= min_cover
    with np.errstate(all='ignore'):
        base = np.nanmean(A[pre], 0)
        out[ok] = (np.nanmean(A[dur], 0) - base)[ok]
    return out, base, cover

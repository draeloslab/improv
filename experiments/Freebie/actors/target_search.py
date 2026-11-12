"""The optimisation inside actors/bayes_opt_targets.py, without improv (so it can be tested offline).

The GP is Code/BayesOptim's (its kernel function; the posterior is computed directly per candidate, which equals its
`model.optimizer.Optimizer` to 1e-12 but does not need the candidates x candidates covariance, too big for the 26k
grid), set up the way the neurostimulation GP-BO papers do it (Markdown/BO_EXPERIMENTS.md has the reasoning and numbers):

  kernel        Matern nu = 5/2 on every input (BayesOptim's `matern`, a product over inputs), as Laferriere 2020,
                Losanno 2021, Bonizzato 2023. One length scale shared by the eight electrode on/off inputs (Laferriere
                bounds a shared spatial length scale), and one each for pulse width, frequency and amplitude (ARD, as
                Bonizzato's variable-length-scale model); `param_length` may also be one number for all three.
  surrogate     one independent GP per output angle (Losanno's "IGP", the model they ran online). The GPs model the
                evoked CHANGE of each output in degrees.
  prior mean    `prior_mean` (candidates x outputs): the GPs learn the residual from it. Built from earlier days with
                prior_from_table(): the posterior mean of the same GPs fitted to the stim table, times `prior_scale`
                (Bonizzato injects a prior scaled to 75 %; Laferriere uses lower-level GPs as the prior mean).
  acquisition   Losanno's online rule for reaching a target vector:
                    x = argmax_x [ -sqrt(sum_i (m_i(x) - target_i)^2) + k * sqrt(sum_i sd_i(x)^2) ]
                over the discrete candidate grid. k is tuned offline on the table (notebooks/bo_walkthrough.ipynb).
  convergence   the same stimulation chosen `converge_repeats` (5) times in a row, as in all three papers; or `max_tests`.
  hyper-params  fixed while running (Losanno: fitted on earlier data, then frozen): fit_hyperparameters().

Additions for the experiments (each off by default):
  outputs       besides the hand_angles names: FINGER_EXT = mean MCP extension of the four fingers (+ = extension) and
                FINGER_SPLIT = index/middle extension minus ring/pinky extension (derived()).
  targets       target_kind 'change' (default): targets are evoked changes. 'posture': targets are absolute angles (zero =
                the warm-up posture); the trial asks for target - current resting posture (experiment C1: the hand is
                released and droops, the posture to reach stays the same).
  regulariser   score -= charge_weight * charge(x) / max charge * |target|          (prefer weaker stimulation)
                score += novelty_weight * (1 - largest correlation with found solutions) * |target|
                with solutions_per_target > 1 the target is searched again after each solution; candidates with the
                electrode set of a found solution (unique: electrodes) or correlated above exclude_corr with one are no
                longer eligible; a solution the GPs put further than accept_deg from the target ends the search of that target.
  fatigue       optimizer.fatigue 'charge' (or 'time'): one more GP input, the session's cumulative charge in mC (or
                minutes since start), Matern with fatigue_length. Every candidate is evaluated "now"; old trials count
                less for the present as the dose since them grows (a time-varying GP, product kernel).
  forgetting    window > 0: the GPs only hold the latest `window` trials.
  cycle         mode 'cycle': the targets are visited in turn for ever, one trial per visit, never finished.

Inputs x: eight electrode on/off values, then pulse width / 300 us, frequency / 100 Hz, amplitude / 6000 uA
(+ the fatigue input when enabled).
"""
import itertools
import sys
import types
from pathlib import Path

import numpy as np

ELECTRODES = [2, 4, 6, 8, 10, 12, 14, 16]
SCALE = np.array([300.0, 100.0, 6000.0])                      # pulse width us, frequency Hz, amplitude uA -> 0..1
FINGERS = ('INDEX', 'MIDDLE', 'RING', 'PINKY')
DEFAULTS = dict(
    bayesoptim_path='~/Code/BayesOptim',
    outputs=['WRIST_EXT', 'WRIST_DEV'],
    targets=[[15, 0], [30, 0], [10, 12], [20, 12], [0, 12]],
    target_names=None,
    target_kind='change',                                     # 'change' | 'posture'
    mode='search',                                            # 'search' | 'cycle'
    space=dict(electrodes=ELECTRODES, max_electrodes=3, pulse_widths=[100, 150, 200], frequencies=[30, 50], currents=[3000, 5000]),
    optimizer=dict(electrode_length=1.5, param_length=0.6, prior_sd=[15.0, 10.0], noise_sd=[4.0, 3.5],
                   k=1.0, converge_repeats=5, max_tests=40, init_random=6, seed=0, window=0,
                   fatigue=None, fatigue_length=20.0, prior_scale=1.0),
    regularizer=dict(charge_weight=0.0, novelty_weight=0.0, solutions_per_target=1, exclude_corr=0.8, unique=None, accept_deg=None),
    timing=dict(stim_duration=1.0, rest_duration=2.0, settle=0.5, baseline=1.0, on_timeout=1.0, off_grace=0.5, eval_timeout=1.0),
    wrist=dict(side='left', warmup_s=10.0),
    prior_table=None, prior_dates=None,
    max_trials=0,
)


def merged(*cfgs):
    """DEFAULTS, then each override; the dict-valued sections are merged key by key."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    for c in cfgs:
        for k, v in (c or {}).items():
            out[k] = {**out[k], **v} if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def encode(electrodes, pulse_width, frequency, amplitude):
    """One stimulation -> the 11 stimulation inputs of the GP."""
    on = np.isin(ELECTRODES, list(electrodes)).astype(float)
    return np.r_[on, np.array([pulse_width, frequency, amplitude], float) / SCALE]


def train_charge_mc(n_electrodes, pulse_width, frequency, amplitude, duration):
    """Charge per phase summed over every pulse of a train, mC (the stim table's charge_nc / 1e6)."""
    return n_electrodes * round(frequency * duration) * amplitude * pulse_width * 1e-9


def derived(a):
    """a: {angle name: value or array}, with the four <F>_MCP_FLEX -> a copy with FINGER_EXT and FINGER_SPLIT added."""
    ext = {f: -np.asarray(a[f'{f}_MCP_FLEX'], float) for f in FINGERS}
    return {**a, 'FINGER_EXT': (ext['INDEX'] + ext['MIDDLE'] + ext['RING'] + ext['PINKY']) / 4.0,
            'FINGER_SPLIT': (ext['INDEX'] + ext['MIDDLE']) / 2.0 - (ext['RING'] + ext['PINKY']) / 2.0}


def table_trials(path, outputs, dates=None, min_duration=0.9):
    """Trains of the stim table (stim_table.csv) whose electrodes share one setting, started together, lasted at least
    min_duration s and were measured -> dict(X (n, 11), Y (n, outputs) deg change, q (n,) cumulative charge of the run
    before the train in mC, date, recording, t_on)."""
    import pandas as pd
    t = pd.read_csv(Path(path).expanduser(), low_memory=False)
    t = t[t.timing_ok & t.on_video & t.timing.isin(['single', 'synchronous']) & (t.duration_s >= min_duration)]
    if dates:
        t = t[t.date.isin(list(dates))]
    d = derived({c[2:]: t[c].to_numpy(float) for c in t.columns if c.startswith('d_')})
    Yall = np.stack([d[o] for o in outputs], 1)
    X, Y, q, keep = [], [], [], []
    for n, (_, r) in enumerate(t.iterrows()):
        els = [int(e) for e in str(r.electrodes).split('-')]
        par = np.array([[r[f'e{e}_w1'], r[f'e{e}_rate'], r[f'e{e}_a1']] for e in els], float)
        if np.isfinite(par).all() and np.allclose(par, par[0]) and np.isfinite(Yall[n]).all():
            X.append(encode(els, *par[0])); Y.append(Yall[n]); q.append((r.cum_charge_run_nc - r.charge_nc) / 1e6); keep.append(n)
    t = t.iloc[keep]
    return dict(X=np.array(X), Y=np.array(Y), q=np.array(q), date=t.date.to_numpy(), recording=t.recording.to_numpy(), t_on=t.t_on.to_numpy(float))


def _bayesoptim(path):
    sys.path.insert(0, str(Path(path).expanduser()))
    from model.kernel import kernel
    from model.optimizer import Optimizer
    return Optimizer, kernel


def gamma_of(o, with_fatigue):
    """BayesOptim's per-input gamma (length = 0.5 / gamma) from the optimizer settings."""
    pl = np.broadcast_to(np.asarray(o['param_length'], float), (3,))
    g = np.r_[np.full(8, 0.5 / float(o['electrode_length'])), 0.5 / pl]
    return np.r_[g, 0.5 / float(o['fatigue_length'])] if with_fatigue else g


def fit_hyperparameters(X, Y, bayesoptim_path='~/Code/BayesOptim', ard=True, fatigue=False, mask=None):
    """Kernel settings from earlier data by maximum marginal likelihood, with BayesOptim's own kernel function.
    X (n, 11) (or (n, 12) with the fatigue input last when fatigue=True), Y (n, outputs) in degrees. Length scales: one
    shared by the electrode inputs (0.5 .. 4), one per parameter if ard else one shared (0.15 .. 3), fatigue (0.5 .. 500);
    per output a prior SD and a noise SD. mask (n, n) bool, optional: pairs allowed to covary (False = independent, e.g.
    trials of different recordings when fitting the fatigue length). Returns a dict in DEFAULTS['optimizer'] form."""
    from scipy.optimize import minimize
    _, kernel = _bayesoptim(bayesoptim_path)
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    kinds = ['matern'] * X.shape[1]
    npar = 3 if ard else 1

    def unit_kernel(th):
        o = dict(electrode_length=np.exp(th[0]), param_length=np.exp(th[1:1 + npar]), fatigue_length=np.exp(th[1 + npar]) if fatigue else 1.0)
        c = types.SimpleNamespace(gamma=gamma_of(o, fatigue), var=1.0, matern_nu=2.5)
        K = kernel(X, X, c, kinds)
        return K * mask if mask is not None else K

    nl = 1 + npar + int(fatigue)

    def nll(th):
        K0, total = unit_kernel(th), 0.0
        for j in range(Y.shape[1]):
            K = np.exp(2 * th[nl + 2 * j]) * K0 + np.exp(2 * th[nl + 1 + 2 * j]) * np.eye(len(X))
            L = np.linalg.cholesky(K + 1e-8 * np.eye(len(X)))
            a = np.linalg.solve(L.T, np.linalg.solve(L, Y[:, j]))
            total += 0.5 * Y[:, j] @ a + np.log(np.diag(L)).sum()
        return total

    sd = Y.std(0)
    x0 = np.r_[np.log(1.5), np.full(npar, np.log(0.6)), [np.log(20.0)] * int(fatigue),
               np.ravel([[np.log(max(s, 1.0)), np.log(max(0.4 * s, 0.5))] for s in sd])]
    bounds = ([(np.log(0.5), np.log(4.0))] + [(np.log(0.15), np.log(3.0))] * npar + [(np.log(0.5), np.log(500.0))] * int(fatigue)
              + [(np.log(0.5), np.log(80.0)), (np.log(0.3), np.log(30.0))] * Y.shape[1])
    r = minimize(nll, x0, method='L-BFGS-B', bounds=bounds, options=dict(maxiter=300))
    pl = np.exp(r.x[1:1 + npar])
    out = dict(electrode_length=float(np.exp(r.x[0])), param_length=[float(v) for v in pl] if ard else float(pl[0]),
               prior_sd=[float(np.exp(r.x[nl + 2 * j])) for j in range(Y.shape[1])],
               noise_sd=[float(np.exp(r.x[nl + 1 + 2 * j])) for j in range(Y.shape[1])], nll=float(r.fun), n=int(len(X)))
    if fatigue:
        out['fatigue_length'] = float(np.exp(r.x[1 + npar]))
    return out


def prior_from_table(cfg, X, Y):
    """Prior mean (candidates x outputs) for TargetSearch(cfg): the posterior mean of the same GPs (no fatigue input)
    fitted to earlier trials X (n, 11), Y (n, outputs), times optimizer.prior_scale."""
    c = merged(cfg, dict(optimizer=dict(fatigue=None, window=0)))
    ts = TargetSearch(c)
    ts.obs_x, ts.obs_y = list(np.asarray(X, float)), list(np.asarray(Y, float))
    ts.obs_prior = [np.zeros(len(ts.outputs))] * len(ts.obs_x)
    ts._initialize()
    return float(cfg['optimizer'].get('prior_scale', 1.0)) * ts.posterior()[0]


class TargetSearch:
    """Candidates, one BayesOptim Optimizer per output angle, and the choice of the next stimulation (module docstring)."""

    def __init__(self, cfg, prior_mean=None):
        cfg = merged(cfg)
        self._Optimizer, self._kernel = _bayesoptim(cfg['bayesoptim_path'])
        sp, self.o, self.reg = cfg['space'], cfg['optimizer'], cfg['regularizer']
        sets = [c for n in range(1, int(sp['max_electrodes']) + 1) for c in itertools.combinations(sp['electrodes'], n)]
        self.pws, self.fqs = list(sp['pulse_widths']), list(sp['frequencies'])
        self.grid = [(c, pw, fq, cu) for c in sets for pw in self.pws for fq in self.fqs for cu in sp['currents']]
        self.X = np.array([encode(*g) for g in self.grid])
        self.outputs = list(cfg['outputs'])
        self.targets = [np.array(t, float) for t in cfg['targets']]
        self.names = cfg.get('target_names') or [str(t) for t in cfg['targets']]
        self.posture = cfg.get('target_kind', 'change') == 'posture'
        self.mode = cfg['mode']
        self.rng = np.random.default_rng(int(self.o['seed']))
        self.fatigue = self.o.get('fatigue')
        self.kernels = ['matern'] * (12 if self.fatigue else 11)
        self.gamma = gamma_of(self.o, bool(self.fatigue))
        self.q_now = 0.0                                       # fatigue input of "now" (cumulative mC or minutes)
        self.prior_mean = np.zeros((len(self.grid), len(self.outputs))) if prior_mean is None else np.asarray(prior_mean, float)
        charge = np.array([len(c) * pw * fq * cu for c, pw, fq, cu in self.grid], float)
        self.charge = charge / charge.max()
        self.optims, self.q_fit = None, None
        self._ks_grid, self._c_stim = [], types.SimpleNamespace(gamma=self.gamma[:11], var=1.0, matern_nu=2.5)
        self.obs_x, self.obs_y, self.obs_prior, self.obs_base = [], [], [], []
        self.active, self.tests, self.streak, self.last_choice, self.tried = 0, 0, 0, None, set()
        self.found = {}                                        # target index -> candidate indices accepted as solutions
        self.dictionary, self.visit = [], 0

    def _cs(self):
        xs = np.c_[self.X, np.full(len(self.X), self.q_now)] if self.fatigue else self.X
        return [types.SimpleNamespace(gamma=self.gamma, var=float(v) ** 2, nu=0.0, eta=float(e), x_star=xs, d=xs.shape[1], matern_nu=2.5)
                for v, e in zip(self.o['prior_sd'], self.o['noise_sd'])]

    def _prior_at(self, x):
        """Prior mean at an arbitrary x (a delivered stimulation need not be a candidate): the nearest candidate's."""
        return self.prior_mean[int(np.argmin(((self.X - np.asarray(x, float)[:11]) ** 2).sum(1)))]

    # -- data in
    def set_fatigue(self, q):
        """The fatigue input of now (cumulative charge in mC, or minutes); the GPs are re-evaluated at it before choose()."""
        self.q_now = float(q)

    def add(self, x, change, q=None, baseline=None):
        """A measured trial: x (11 inputs), change (one value per output, deg), q its fatigue input (cumulative charge
        before the train), baseline the resting posture before it (posture targets). False if not fully measured."""
        change = np.asarray(change, float)
        if not np.isfinite(change).all():
            return False
        x = np.asarray(x, float)[:11]
        self.obs_x.append(np.r_[x, self.q_now if q is None else float(q)] if self.fatigue else x)
        self.obs_y.append(change); self.obs_prior.append(self._prior_at(x))
        self.obs_base.append(np.zeros(len(change)) if baseline is None else np.asarray(baseline, float))
        if self.optims is not None:
            self.q_fit = None                                  # the posterior is recomputed at the next posterior()
        elif len(self.obs_x) >= max(int(self.o['init_random']), 1):
            self._initialize()
        return True

    def _initialize(self):
        """The GPs hold data from now on (the posterior itself is computed on demand in posterior())."""
        self.optims, self.q_fit = [], None

    def _direct_posterior(self):
        """Posterior mean and variance of every candidate (at the fatigue input of now, when there is one). Same GP and
        kernel (BayesOptim's kernel function) as BayesOptim's Optimizer, to 1e-12 (tests), but only the candidates'
        variances are computed: Optimizer.initialize_GP builds the full candidates x candidates covariance, which does not
        fit in memory for the 26k-candidate grid (5.5 GB) and has to be rebuilt whenever "now" moves (~9 s live for 1656
        candidates). The stimulation part of the kernel, k_stim(trial, candidate), is cached as trials arrive; with a
        fatigue input it is scaled by k_q(q_trial, q_now) at "now"."""
        n_all, n0 = len(self.obs_x), len(self._ks_grid)
        if n0 < n_all:                                         # cache the rows of the new trials
            self._ks_grid.extend(self._kernel(np.array(self.obs_x[n0:])[:, :11], self.X, self._c_stim, ['matern'] * 11))
        w = int(self.o['window']) or n_all
        Xo = np.array(self.obs_x[-w:]); R = np.array(self.obs_y[-w:]) - np.array(self.obs_prior[-w:])
        Koo = self._kernel(Xo[:, :11], Xo[:, :11], self._c_stim, ['matern'] * 11)
        Kog = np.array(self._ks_grid[-w:])                                                                   # (n, V)
        if self.fatigue:
            q = Xo[:, 11:12]
            cq = types.SimpleNamespace(gamma=self.gamma[11:], var=1.0, matern_nu=2.5)
            Koo = Koo * self._kernel(q, q, cq, ['matern'])
            Kog = Kog * self._kernel(q, np.array([[self.q_now]]), cq, ['matern'])
        f, v = [], []
        for j, (sd0, eta) in enumerate(zip(self.o['prior_sd'], self.o['noise_sd'])):
            L = np.linalg.cholesky(sd0 ** 2 * Koo + eta ** 2 * np.eye(len(Xo)))
            a = np.linalg.solve(L.T, np.linalg.solve(L, R[:, j]))
            B = np.linalg.solve(L, sd0 ** 2 * Kog)
            f.append(sd0 ** 2 * Kog.T @ a); v.append(sd0 ** 2 - (B ** 2).sum(0))
        self._fp, self.q_fit = (np.stack(f, 1), np.stack(v, 1)), self.q_now

    def posterior(self):
        """Predicted change (candidates x outputs) and its SD, now."""
        if self.q_fit != self.q_now:
            self._direct_posterior()
        f, var = self._fp
        return self.prior_mean + f, np.sqrt(np.maximum(var, 0.0))

    def miss_slice(self, electrodes, current, wanted):
        """Predicted distance (deg) to the wanted change over pulse width x frequency, for one electrode set and current
        (the live GUI's heatmap). Returns an array of shape (len(pws), len(fqs))."""
        m = self.posterior()[0] if self.optims is not None else self.prior_mean
        sel = [i for i, g in enumerate(self.grid) if g[0] == tuple(electrodes) and g[3] == current]      # pulse width, then frequency
        return np.linalg.norm(m[sel] - np.asarray(wanted, float), axis=1).reshape(len(self.pws), len(self.fqs))

    # -- next stimulation
    def current_target(self):
        if self.mode == 'cycle':
            return self.visit % len(self.targets)
        return self.active if self.active < len(self.targets) else None

    def wanted_change(self, t, baseline=None):
        """The change asked of the next trial for target t: the target, or (posture targets) target - resting posture."""
        tau = self.targets[t]
        return tau - np.asarray(baseline, float) if self.posture and baseline is not None else tau

    def _excluded(self, t):
        found = self.found.get(t, [])
        if not found:
            return np.zeros(len(self.grid), bool), np.zeros(len(self.grid))
        near = self._kernel(self.X, self.X[found], self._c_stim, ['matern'] * 11).max(1)   # correlation with the nearest solution
        out = near > float(self.reg['exclude_corr'])
        if self.reg.get('unique') == 'electrodes':
            sets = {self.grid[j][0] for j in found}
            out |= np.array([g[0] in sets for g in self.grid])
        return out, near

    def choose(self, baseline=None):
        """(candidate index, info) for the current target, or (None, ...) when every target is finished. baseline: the
        resting posture now (outputs, deg), used by posture targets."""
        t = self.current_target()
        if t is None:
            return None, dict(mode='finished')
        tau = self.wanted_change(t, baseline)
        excluded, near = self._excluded(t)
        if self.optims is None and int(self.o['init_random']) > 0:
            ok = np.flatnonzero(~excluded)
            return int(self.rng.choice(ok)), dict(mode='initial random', target=self.targets[t].tolist(), wanted=tau.tolist(), name=self.names[t])
        if self.optims is None:                 # no trial yet, init_random 0: the prior mean and the prior SD
            m, sd = self.prior_mean, np.broadcast_to(np.asarray(self.o['prior_sd'], float), self.prior_mean.shape)
        else:
            m, sd = self.posterior()
        dist, spread = np.linalg.norm(m - tau, axis=1), np.sqrt((sd ** 2).sum(1))
        size = np.linalg.norm(tau)
        score = -dist + float(self.o['k']) * spread - float(self.reg['charge_weight']) * self.charge * size
        score = score + float(self.reg['novelty_weight']) * (1.0 - near) * size
        score[excluded] = -np.inf
        i = int(np.argmax(score))
        if not np.isfinite(score[i]):
            return None, dict(mode='no eligible candidate', target=self.targets[t].tolist())
        return i, dict(mode='losanno-ucb', target=self.targets[t].tolist(), wanted=tau.tolist(), name=self.names[t], score=float(score[i]),
                       predicted=m[i].tolist(), sd=sd[i].tolist(), predicted_distance=float(dist[i]), charge=float(self.charge[i]), q=self.q_now)

    def finish_trial(self, i, change):
        """After add(): book the trial; in search mode, close the target when it has converged. Returns why, or None."""
        t = self.current_target()
        if t is None:
            return None
        if self.mode == 'cycle':
            self.visit += 1
            return None
        if self.optims is None:
            return None
        self.tests += 1
        self.tried.add(int(i))
        self.streak = self.streak + 1 if i == self.last_choice else 1
        self.last_choice = i
        why = ('converged' if self.streak >= int(self.o['converge_repeats']) else
               'max tests' if self.tests >= int(self.o['max_tests']) else None)
        if why:
            self._book(t, i, why)
        return why

    def give_up(self):
        """End the current target now (e.g. no eligible candidate left): book the best tried stimulation."""
        t = self.current_target()
        if t is not None and self.mode != 'cycle':
            self._book(t, None, 'stopped')

    def _book(self, t, i, why):
        tau = self.targets[t]
        tried = sorted(self.tried)
        if why != 'converged' and not tried:
            self.active += 1
            return
        # posture targets: the measured posture of a trial is resting posture + change
        meas_of = [(x, y + b) if self.posture else (x, y) for x, y, b in zip(self.obs_x, self.obs_y, self.obs_base)]
        m, sd = self.posterior()
        if why == 'converged':
            j = i
        else:                                   # not converged: the tested stimulation the GPs now put nearest the target
            want = self.wanted_change(t, self.obs_base[-1] if self.obs_base else None)     # (picking the best single measurement would flatter it)
            j = tried[int(np.argmin(np.linalg.norm(m[tried] - want, axis=1)))]
        same = [y for x, y in meas_of if np.allclose(x[:11], self.X[j])]
        meas = np.mean(same, 0) if same else np.full(len(tau), np.nan)
        dist = float(np.linalg.norm(meas - tau)) if same else None
        acc = self.reg.get('accept_deg')
        pred_dist = float(np.linalg.norm(m[j] - self.wanted_change(t, self.obs_base[-1] if self.obs_base else None)))
        accepted = acc is None or pred_dist <= float(acc)    # the GP's estimate: a 1-2 repeat measurement is mostly noise
        c, pw, fq, cu = self.grid[j]
        self.dictionary.append(dict(target=tau.tolist(), name=self.names[t], outputs=self.outputs, kind='posture' if self.posture else 'change',
                                    solution=len(self.found.get(t, [])) + 1, finished=why, accepted=accepted,
                                    trials=self.tests, electrodes=[int(e) for e in c], pulse_width=int(pw), frequency=int(fq), amplitude=int(cu),
                                    charge=float(self.charge[j]), predicted=m[j].tolist(), predicted_sd=sd[j].tolist(),
                                    measured=meas.tolist(), repeats=len(same), distance_deg=dist, predicted_distance_deg=pred_dist))
        self.found.setdefault(t, []).append(j)
        self.tests, self.streak, self.last_choice, self.tried = 0, 0, None, set()
        if len(self.found[t]) >= int(self.reg['solutions_per_target']) or not accepted:
            self.active += 1

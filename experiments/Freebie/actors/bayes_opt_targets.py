"""Bayesian optimisation of the stimulation towards target postures, live (the search itself is actors/target_search.py).

Same wiring and trial state machine as actors/bayes_opt.py (request -> BRAND stim_on / stim_off -> measure -> update):

    ProcessorBatch3D.q_out ──> joints_in ┐
                                         ├─ BayesOptTargets ── q_out ──> SenderUDP.stim_in ──> BRAND (stim port)
    BrandReceiver.q_out ───> feedback_in ┘

GP and acquisition: one GP per output angle with Code/BayesOptim's kernel (posterior computed directly per candidate,
= its Optimizer to 1e-12), Matern 5/2, Losanno's distance-to-target UCB,
convergence on the same stimulation five times in a row (actors/target_search.py; Markdown/BO_EXPERIMENTS.md).

Settings: config/config.yaml block `bayes_opt_targets`, overridden by the block the graph names with `settings:`
(bo_exp_A, bo_exp_B, bo_exp_C1, bo_exp_C2), e.g.  BayesOptTargets: {package: ..., settings: bo_exp_A}

Measured outcome of a trial: the CHANGE of every output angle in degrees, mean over [on + settle, off) minus mean over
the `baseline` seconds before on (actors.stim_gp.window_change). The angles come from the processor's live 3D keypoints
with hand_training/hand_angles.py: WRIST_EXT, WRIST_DEV, <FINGER>_MCP_FLEX, and FINGER_EXT / FINGER_SPLIT
(target_search.derived). The wrist is one fixed point and the zero posture the hand's usual direction, both from the
first `warmup_s` seconds (the forearm is held); nothing is requested before then.
Every trial is appended to <run folder>/bo_trials.jsonl (with the resting posture, the fatigue input and the predicted
change); finished targets go to bo_dictionary.json; the GP data to bo_gp_data.npz at stop.
"""
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

from . import cpu_affinity
from .bayes_opt import BayesOptStim
from .run_paths import get_logger, run_folder
from .stim_gp import window_change
from .target_search import TargetSearch, derived, encode, merged, prior_from_table, table_trials, train_charge_mc

logger = get_logger(__name__, "bayes_opt_targets.log")


class BayesOptTargets(BayesOptStim):
    """BayesOptStim's trial state machine with BayesOptim's Optimizer and posture targets (module docstring)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings = kwargs.get('settings')              # config.yaml block overriding bayes_opt_targets

    def setup(self):
        cpu_affinity.pin_actor(cpu_affinity.BACKGROUND, label="BayesOptTargets")
        src = Path(__file__).resolve().parent.parent
        config = yaml.safe_load((src / 'config' / 'config.yaml').read_text()) or {}
        over = config.get(self.settings) if self.settings else None
        if self.settings and over is None:
            raise KeyError(f"config.yaml has no block '{self.settings}'")
        cfg = merged(config.get('bayes_opt_targets'), over)
        self.cfg, self.T = cfg, cfg['timing']
        self.joints = list(cfg['outputs'])
        sys.path.insert(0, str(src / 'hand_training'))
        import hand_angles
        self.ha = hand_angles
        prior = None
        if cfg['prior_table']:
            d = table_trials(cfg['prior_table'], self.joints, cfg.get('prior_dates'))
            prior = prior_from_table(cfg, d['X'], d['Y'])
            logger.info(f"prior mean from {len(d['X'])} table trains ({sorted(set(d['date']))}), scale {cfg['optimizer']['prior_scale']}")
        self.search = TargetSearch(cfg, prior)
        self.q_mc, self.t_start = 0.0, None                 # cumulative charge delivered this session (mC); time of the first request
        self.warm, self.t_first, self.W, self.f = [], None, None, None
        self.buf = deque(maxlen=60 * 30)
        self.target = True                          # the parent's idle check wants a target; ours are in the config
        self.state, self.trial, self.n_trials = 'idle', None, 0
        self.next_allowed = time.time() + 2.0
        self.req_id = int(time.time()) % 100000 * 1000
        self.refit, self.trials, self.on_latency = None, [], deque(maxlen=20)
        self.out_folder = run_folder()
        self.log_path = self.out_folder / 'bo_trials.jsonl'
        (self.out_folder / 'bo_settings.json').write_text(json.dumps(dict(settings=self.settings, cfg=cfg), indent=1, default=str))
        logger.info(f"settings {self.settings}: {len(self.search.grid)} candidates, outputs {self.joints}, targets {cfg['targets']} "
                    f"({cfg['target_kind']}), mode {cfg['mode']}, optimizer {cfg['optimizer']}, regularizer {cfg['regularizer']}; trials -> {self.log_path}")

    def _set_target(self, m):
        pass                                        # the targets are the config's list; BRAND's task target is not used here

    def _maybe_refit(self):
        pass                                        # the hyper-parameters are fixed while running (fitted beforehand)

    def _take_joints(self):
        """The output angles of the newest frames, from the processor's 3D keypoints."""
        for msg in self._drain('joints_in'):
            if not isinstance(msg, dict) or msg.get('points_3d') is None:
                continue
            p = np.asarray(msg['points_3d'], float).reshape(2, 21, 3)
            hand = p[int(np.isfinite(p[1, :, 0]).sum() >= np.isfinite(p[0, :, 0]).sum())][None]      # the one hand, whichever block holds it
            t = float(msg.get('camera_start') or time.time())
            if self.f is None:                      # warm-up: where the wrist is and where the hand usually points
                self.t_first = self.t_first or t
                self.warm.append(hand[0])
                if t - self.t_first >= float(self.cfg['wrist']['warmup_s']):
                    P = np.array(self.warm)
                    w = P[:, 0][np.isfinite(P[:, 0, 0])]
                    if len(w) >= 30:
                        self.W = np.median(w, 0)
                        self.f = self.ha.hand_axis(self.ha.with_wrist(P, self.W), self.cfg['wrist']['side'])
                        logger.info(f"wrist point {np.round(self.W, 1)} mm from {len(w)} frames; reference axis {np.round(self.f, 3)}")
                    self.warm = self.warm[-600:]
                continue
            side = self.cfg['wrist']['side']
            Q = self.ha.with_wrist(hand, self.W)
            flex, dev = self.ha.wrist_angles(Q, self.f, side)
            a = {'WRIST_EXT': -flex[0], 'WRIST_DEV': dev[0]}
            if any(j not in a for j in self.joints):
                a = derived({**a, **{k: v[0] for k, v in self.ha.ray_angles(Q, side).items()}})
            self.buf.append((t, np.array([a[j] for j in self.joints], float)))

    def _resting(self, now):
        """Mean of each output over the last `baseline` seconds (the posture a posture target starts from)."""
        if not self.buf:
            return None
        t = np.array([b[0] for b in self.buf]); A = np.array([b[1] for b in self.buf])
        sel = t >= t[-1] - float(self.T['baseline'])
        with np.errstate(invalid='ignore'):
            r = np.nanmean(A[sel], 0) if sel.any() else np.full(len(self.joints), np.nan)
        return r if np.isfinite(r).all() else None

    def _fatigue_now(self, now):
        kind = self.cfg['optimizer'].get('fatigue')
        return self.q_mc if kind == 'charge' else (now - self.t_start) / 60.0 if kind == 'time' else 0.0

    def _request(self, now):
        if self.f is None:
            return
        max_t = int(self.cfg['max_trials'])
        if max_t and self.n_trials >= max_t:
            return
        self.t_start = self.t_start or now
        base = self._resting(now)
        if self.search.posture and base is None:
            return                                  # posture targets need the resting posture: wait for tracked frames
        q = self._fatigue_now(now)
        if abs(q - self.search.q_now) > 0.1 * float(self.cfg['optimizer']['fatigue_length']) or self.cfg['optimizer'].get('fatigue') == 'charge':
            self.search.set_fatigue(q)              # (time: the GPs were re-fitted during the rest; move "now" only when it has moved)
        i, why = self.search.choose(base)
        if i is None and why.get('mode') == 'no eligible candidate':
            self.search.give_up()
            logger.info(f"no eligible candidate left: {self.search.dictionary[-1] if self.search.dictionary else ''}")
            i, why = self.search.choose(base)
        if i is None:
            return
        c, pw, fq, cu = self.search.grid[i]
        self.req_id += 1
        req = dict(type='stim_request', id=self.req_id, electrodes=[int(e) for e in c], amplitude=int(cu), pulse_width=int(pw),
                   frequency=int(fq), duration=float(self.T['stim_duration']), t_sent=now)
        self.trial = dict(id=self.req_id, request=req, t_sent=now, acquisition=why, candidate=i, q=self.search.q_now,
                          resting=None if base is None else base.tolist(), target_index=self.search.current_target())
        self.q_out.put(req)
        self.state = 'waiting_on'
        logger.info(f"trial {self.n_trials + 1}: request {self.req_id} electrodes {req['electrodes']} {req['amplitude']} uA / "
                    f"{req['pulse_width']} us / {req['frequency']} Hz (target {why.get('name')}, {why['mode']})")

    def _evaluate(self, now):
        tr = self.trial
        t = np.array([b[0] for b in self.buf]); A = np.array([b[1] for b in self.buf])
        change, base, cover = window_change(t, A, tr['t_on'], tr['t_off'], self.T['settle'], self.T['baseline'])
        tr['change'] = {j: (None if not np.isfinite(v) else float(v)) for j, v in zip(self.joints, change)}
        tr['baseline'] = {j: (None if not np.isfinite(v) else float(v)) for j, v in zip(self.joints, base)}
        d, r = tr.get('delivered') or {}, tr['request']
        el, pw, fq, cu = (d.get('electrodes') or r['electrodes'], d.get('pulse_width') or r['pulse_width'],
                          d.get('frequency') or r['frequency'], d.get('amplitude') or r['amplitude'])   # what BRAND actually delivered
        dur = float(d.get('duration') or (tr['t_off'] - tr['t_on']))
        tr['charge_mc'] = train_charge_mc(len(el), pw, fq, cu, dur)
        measured = self.search.add(encode(el, pw, fq, cu), change, q=tr['q'], baseline=base)
        self.q_mc += tr['charge_mc']                # the muscles got it whether or not the angles were measured
        done = self.search.finish_trial(tr['candidate'], change) if measured else None
        if self.search.optims is not None:
            t0 = time.time()                        # re-fit the GPs at the new "now" during the rest, not before the next request
            self.search.set_fatigue(self._fatigue_now(now)); self.search.posterior()
            tr['refit_s'] = time.time() - t0
        if done:
            logger.info(f"target finished ({done}): {self.search.dictionary[-1]}")
            (self.out_folder / 'bo_dictionary.json').write_text(json.dumps(self.search.dictionary, indent=1))
        self._close('ok' if measured else 'angles not measured', now)

    def stop(self):
        logger.info(f"BayesOptTargets stopping after {self.n_trials} trials, {self.q_mc:.1f} mC; {len(self.search.dictionary)} dictionary entries")
        (self.out_folder / 'bo_dictionary.json').write_text(json.dumps(self.search.dictionary, indent=1))
        s = self.search
        if s.obs_x:
            np.savez_compressed(self.out_folder / 'bo_gp_data.npz', outputs=np.array(s.outputs), X=np.array(s.obs_x), Y=np.array(s.obs_y),
                                prior=np.array(s.obs_prior), resting=np.array(s.obs_base))

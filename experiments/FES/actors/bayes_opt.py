"""Closed-loop Bayesian optimisation of the stimulation: joint angles in, next stimulation out.

    ProcessorBatch3D.q_out ──> joints_in ┐
                                         ├─ BayesOptStim ── q_out ──> SenderUDP.stim_in ──> BRAND (stim port)
    BrandReceiver.q_out ───> feedback_in ┘

One trial = choose -> request -> BRAND stimulates -> measure -> update:
  1. choose the candidate stimulation (electrode set x pulse width x frequency x current, config
     `bayes_opt.space`) with the best acquisition score, from per-joint GPs (actors/stim_gp.py);
  2. send it as a stim_request (SenderUDP forwards it to BRAND);
  3. BRAND reports stim_on / stim_off (or stim_rejected); their arrival times are this machine's clock,
     the same clock as the processor's frame times (camera_start);
  4. once frames past stim_off have arrived: change = mean(joint angles over [on + settle, off)) minus
     mean(over [on - baseline, on)), the analysis notebook's definition; add it to the GPs;
  5. wait rest_duration after stim_off, then the next trial.
The hyper-parameters are refitted every `refit_every` trials in a background thread; in between, a new
observation only updates the Cholesky factor.

Acquisition (`bayes_opt.acquisition.mode`):
  target   reach BRAND's current target posture. For each candidate the predicted posture is the current
           resting posture (last `baseline` s) + the GP's predicted change; d = predicted - target per joint.
           Score = E[|d|^2] - kappa * SD[|d|^2] (a lower confidence bound on the squared error); lowest wins.
  explore  most informative candidate, sum_j 0.5 log(1 + sd_j^2 / noise_j^2) (as in the GP notebook).

Every trial is appended to <run folder>/bo_trials.jsonl as it finishes (crash-safe).
"""
import json
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from improv.actor import Actor

from . import cpu_affinity
from .run_paths import get_logger, run_folder
from .stim_gp import StimModel, load_prior_runs, window_change

logger = get_logger(__name__, "bayes_opt.log")

DEFAULTS = dict(
    joints=['left_INDEX_MCP', 'left_MIDDLE_MCP', 'left_RING_MCP', 'left_PINKY_MCP'],
    prior_runs=[], prior_joint_prefix='left_', refit_every=10,
    space=dict(electrodes=[2, 4, 6, 8, 10, 12, 14, 16], pulse_widths=[80, 100, 120, 160, 200],
               frequencies=[20, 30, 50, 100], currents=[1000, 2000, 3000, 4000, 5000], max_electrodes=8),
    timing=dict(stim_duration=1.5, rest_duration=2.5, settle=0.5, baseline=1.0, on_timeout=1.0, off_grace=0.5,
                eval_timeout=1.0),
    acquisition=dict(mode='target', kappa=1.0),
    finger_map={},
    max_trials=0,
)


class BayesOptStim(Actor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ setup
    def setup(self):
        cpu_affinity.pin_actor(cpu_affinity.BACKGROUND, label="BayesOptStim")
        source_folder = Path(__file__).resolve().parent.parent
        with open(source_folder / 'config' / 'config.yaml') as f:
            config = yaml.safe_load(f) or {}
        cfg = {**DEFAULTS, **(config.get('bayes_opt') or {})}
        for k in ('space', 'timing', 'acquisition'):
            cfg[k] = {**DEFAULTS[k], **(cfg.get(k) or {})}
        self.cfg = cfg
        self.joints = list(cfg['joints'])
        self.T = cfg['timing']
        sp = cfg['space']
        self.model = StimModel(self.joints, sp['electrodes'], sp['pulse_widths'], sp['frequencies'],
                               sp['currents'], sp['max_electrodes'])
        prior = load_prior_runs([p if Path(p).is_absolute() else source_folder / p for p in cfg['prior_runs']],
                                cfg['prior_joint_prefix'])
        self.model.add_table(prior)
        t0 = time.perf_counter()
        self.model.fit(starts=3)
        logger.info(f"{len(self.model.grid)} candidates; {len(prior)} prior stimulations {self.model.n_obs()}; "
                    f"fit {time.perf_counter() - t0:.1f} s; joints {self.joints}")
        self._refresh_predictions()

        self.buf = deque(maxlen=60 * 30)            # (frame time, joint angles) for the last ~60 s
        self.target = None                          # {joint: absolute deg}
        self.state, self.trial, self.n_trials = 'idle', None, 0
        self.next_allowed = time.time() + 2.0       # let frames and the target arrive first
        self.req_id = int(time.time()) % 100000 * 1000
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='gp-refit')
        self.refit = None
        self.trials = []
        self.on_latency = deque(maxlen=20)        # request -> stim_on (s), to send the next request that much early
        self.out_folder = run_folder()
        self.log_path = self.out_folder / 'bo_trials.jsonl'
        logger.info(f"acquisition {cfg['acquisition']}, timing {self.T}; trials -> {self.log_path}")

    # ------------------------------------------------------------------ inputs
    def _drain(self, name):
        link = self.links.get(name)
        out = []
        if link is None:
            return out
        while True:
            try:
                out.append(link.get_nowait())
            except Exception:
                return out

    def _take_joints(self):
        for msg in self._drain('joints_in'):
            if not isinstance(msg, dict) or not msg.get('joint_angles'):
                continue
            a = msg['joint_angles']
            self.buf.append((float(msg.get('camera_start') or time.time()),
                             np.array([a.get(j, np.nan) if a.get(j) is not None else np.nan for j in self.joints], float)))

    def _take_feedback(self):
        for m in self._drain('feedback_in'):
            kind = m.get('type')
            if kind == 'target':
                self._set_target(m)
            elif self.trial is not None and m.get('id') == self.trial['id']:
                if kind == 'stim_on':
                    self.trial.update(t_on=m['t_rx'], delivered={k: m.get(k) for k in
                                      ('electrodes', 'amplitude', 'pulse_width', 'frequency', 'duration')},
                                      t_on_brand=m.get('t_brand'))
                elif kind == 'stim_off':
                    self.trial.update(t_off=m['t_rx'], t_off_brand=m.get('t_brand'))
                elif kind == 'stim_rejected':
                    self.trial.update(rejected=m.get('reason', ''))
            elif kind in ('stim_on', 'stim_off', 'stim_rejected'):
                logger.warning(f"feedback for request {m.get('id')} but the open trial is "
                               f"{None if self.trial is None else self.trial['id']}: {m}")

    def _set_target(self, m):
        """BRAND's target -> absolute joint angles. Accepts {"joint_angles": {name: deg}} directly, or
        {"fingers": {finger: 0..1}} mapped through config bayes_opt.finger_map
        (finger: {joints: [...], straight: deg, flexed: deg})."""
        tgt = {}
        for j, v in (m.get('joint_angles') or {}).items():
            if j in self.joints and v is not None:
                tgt[j] = float(v)
        for finger, frac in (m.get('fingers') or {}).items():
            fm = self.cfg['finger_map'].get(finger)
            if fm is None or frac is None:
                continue
            for j in fm['joints']:
                if j in self.joints:
                    tgt[j] = fm['straight'] + float(frac) * (fm['flexed'] - fm['straight'])
        if tgt and tgt != self.target:
            logger.info(f"target: {tgt}")
        self.target = tgt or self.target

    # ------------------------------------------------------------------ model
    def _refresh_predictions(self):
        t0 = time.perf_counter()
        self.pred_mu, self.pred_sd = self.model.predict_grid()
        self.noise = np.array([self.model.gps[j].noise_deg for j in self.joints])
        logger.debug(f"grid predictions {time.perf_counter() - t0:.2f} s")

    def _posture_now(self):
        if not self.buf:
            return None
        t_last = self.buf[-1][0]
        A = np.array([a for t, a in self.buf if t >= t_last - self.T['baseline']])
        with np.errstate(all='ignore'):
            return np.nanmean(A, 0)

    def _choose(self):
        mode = self.cfg['acquisition']['mode']
        mu, sd = self.pred_mu, self.pred_sd
        if mode == 'target' and self.target:
            base = self._posture_now()
            cols = [i for i, j in enumerate(self.joints) if j in self.target and base is not None and np.isfinite(base[i])]
            if cols:
                tau = np.array([self.target[self.joints[i]] for i in cols])
                m = base[cols] + mu[:, cols] - tau
                s2 = sd[:, cols] ** 2
                e = (m ** 2 + s2).sum(1)
                v = (2 * s2 ** 2 + 4 * m ** 2 * s2).sum(1)
                score = e - float(self.cfg['acquisition'].get('kappa', 1.0)) * np.sqrt(v)
                i = int(np.argmin(score))
                return i, dict(mode='target', score=float(score[i]), expected_rms_deg=float(np.sqrt(e[i] / len(cols))),
                               target={self.joints[c]: float(t) for c, t in zip(cols, tau)},
                               baseline={self.joints[c]: float(base[c]) for c in cols})
        info = 0.5 * np.log1p(sd ** 2 / self.noise ** 2).sum(1)
        i = int(np.argmax(info))
        return i, dict(mode='explore', score=float(info[i]))

    def _maybe_refit(self):
        if self.refit is not None and self.refit.done():
            try:
                thetas = self.refit.result()
                for j, (th, mu, sd) in thetas.items():
                    g = self.model.gps[j]; g.th, g.mu, g.sd = th, mu, sd; g._factor()
                self._refresh_predictions()
                logger.info(f"hyper-parameters refitted after {self.n_trials} trials")
            except Exception:
                logger.error(f"refit failed: {traceback.format_exc()}")
            self.refit = None
        every = int(self.cfg['refit_every'])
        if self.refit is None and every > 0 and self.n_trials and self.n_trials % every == 0 \
                and getattr(self, '_last_refit', -1) != self.n_trials:
            self._last_refit = self.n_trials
            snap = {j: (g.E.copy(), g.Z.copy(), g.y.copy(), g.th.copy()) for j, g in self.model.gps.items()}

            def work():
                from .stim_gp import JointGP
                out = {}
                for j, (E, Z, y, th) in snap.items():
                    g = JointGP(); g.E, g.Z, g.y, g.th = E, Z, y, th
                    g.fit(starts=2)
                    out[j] = (g.th, g.mu, g.sd)
                return out
            self.refit = self.pool.submit(work)

    # ------------------------------------------------------------------ loop
    def runStep(self):
        try:
            self._take_joints()
            self._take_feedback()
            self._maybe_refit()
            now = time.time()
            if self.state == 'idle':
                if now >= self.next_allowed and self.buf and (self.target or self.cfg['acquisition']['mode'] != 'target'):
                    self._request(now)
            elif self.state == 'waiting_on':
                if self.trial.get('rejected') is not None:
                    self._close('rejected', now)
                elif 't_on' in self.trial:
                    self.on_latency.append(self.trial['t_on'] - self.trial['t_sent'])
                    self.state = 'stimulating'
                elif now - self.trial['t_sent'] > self.T['on_timeout']:
                    self._close('no stim_on from BRAND', now)
            elif self.state == 'stimulating':
                if 't_off' not in self.trial and now > self.trial['t_on'] + self.T['stim_duration'] + self.T['off_grace']:
                    self.trial['t_off'] = self.trial['t_on'] + self.T['stim_duration']      # no stim_off: assume nominal
                    self.trial['t_off_assumed'] = True
                if 't_off' in self.trial:
                    latest = self.buf[-1][0] if self.buf else -np.inf
                    if latest >= self.trial['t_off'] or now > self.trial['t_off'] + self.T['eval_timeout']:
                        self._evaluate(now)
        except Exception:
            logger.error(f"runStep: {traceback.format_exc()}")

    def _request(self, now):
        max_t = int(self.cfg['max_trials'])
        if max_t and self.n_trials >= max_t:
            return
        i, why = self._choose()
        c, pw, fq, cu = self.model.grid[i]
        self.req_id += 1
        req = dict(type='stim_request', id=self.req_id, electrodes=[int(e) for e in c], amplitude=int(cu),
                   pulse_width=int(pw), frequency=int(fq), duration=float(self.T['stim_duration']), t_sent=now)
        pred = {j: [float(self.pred_mu[i, k]), float(self.pred_sd[i, k])] for k, j in enumerate(self.joints)}
        self.trial = dict(id=self.req_id, request=req, t_sent=now, acquisition=why, predicted=pred)
        self.q_out.put(req)
        self.state = 'waiting_on'
        logger.info(f"trial {self.n_trials + 1}: request {self.req_id} electrodes {req['electrodes']} "
                    f"{req['amplitude']} uA / {req['pulse_width']} us / {req['frequency']} Hz ({why['mode']}, score {why['score']:.2f})")

    def _evaluate(self, now):
        tr = self.trial
        t = np.array([b[0] for b in self.buf]); A = np.array([b[1] for b in self.buf])
        change, base, cover = window_change(t, A, tr['t_on'], tr['t_off'], self.T['settle'], self.T['baseline'])
        tr['change'] = {j: (None if not np.isfinite(v) else float(v)) for j, v in zip(self.joints, change)}
        tr['baseline'] = {j: (None if not np.isfinite(v) else float(v)) for j, v in zip(self.joints, base)}
        tr['cover'] = {j: float(v) for j, v in zip(self.joints, cover)}
        d = tr.get('delivered') or {}
        r = tr['request']
        # the GP learns what was actually delivered (BRAND may clamp), falling back to the request
        el = d.get('electrodes') or r['electrodes']
        pw, fq, cu = (d.get('pulse_width') or r['pulse_width'], d.get('frequency') or r['frequency'],
                      d.get('amplitude') or r['amplitude'])
        if np.isfinite(change).any():
            self.model.add(el, pw, fq, cu, dict(zip(self.joints, change)))
            self._refresh_predictions()
        self._close('ok' if np.isfinite(change).any() else 'no joints measured', now)

    def _close(self, status, now):
        tr = self.trial
        tr['status'] = status
        tr['t_closed'] = now
        self.trials.append(tr)
        self.n_trials += 1
        try:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(tr, default=float) + '\n')
        except Exception:
            logger.error(f"could not append trial: {traceback.format_exc()}")
        msg = ', '.join(f"{j.split('_', 1)[-1]} {v:+.1f}" for j, v in (tr.get('change') or {}).items() if v is not None)
        logger.info(f"trial {self.n_trials} [{status}] {msg}")
        lead = min(max(float(np.median(self.on_latency)), 0.0), 0.5) if self.on_latency else 0.0
        self.next_allowed = (tr.get('t_off') or now) + self.T['rest_duration'] - lead   # the NEXT train starts rest_duration after this one
        self.state, self.trial = 'idle', None

    def stop(self):
        logger.info(f"BayesOptStim stopping after {self.n_trials} trials")
        try:
            self.pool.shutdown(wait=False, cancel_futures=True)
            np.savez_compressed(self.out_folder / 'bo_gp_data.npz', joints=np.array(self.joints),
                                **{f'{j}_E': g.E for j, g in self.model.gps.items()},
                                **{f'{j}_Z': g.Z for j, g in self.model.gps.items()},
                                **{f'{j}_y': g.y for j, g in self.model.gps.items()},
                                **{f'{j}_theta': g.th for j, g in self.model.gps.items()})
        except Exception:
            logger.error(f"could not save GP data: {traceback.format_exc()}")

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
from .target_search import TargetSearch, charge_setting, derived, merged, prior_from_table, table_trials, train_charge_mc

logger = get_logger(__name__, "bayes_opt_targets.log")


class BayesOptTargets(BayesOptStim):
    """BayesOptStim's trial state machine with BayesOptim's Optimizer and posture targets (module docstring)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings = kwargs.get('settings')              # config.yaml block overriding bayes_opt_targets
        self.settings_file = kwargs.get('settings_file')    # optional YAML overriding both (written by scripts/monday.py)

    def setup(self):
        cpu_affinity.pin_actor(cpu_affinity.BACKGROUND, label="BayesOptTargets")
        src = Path(__file__).resolve().parent.parent
        config = yaml.safe_load((src / 'config' / 'config.yaml').read_text()) or {}
        over = config.get(self.settings) if self.settings else None
        if self.settings and over is None:
            raise KeyError(f"config.yaml has no block '{self.settings}'")
        over2 = yaml.safe_load(Path(self.settings_file).expanduser().read_text()) if self.settings_file else None
        cfg = merged(config.get('bayes_opt_targets'), over, over2)
        self.cfg, self.T = cfg, cfg['timing']
        self.joints = list(cfg['outputs'])
        sys.path.insert(0, str(src / 'hand_training'))
        import hand_angles
        self.ha = hand_angles
        prior = None
        if cfg['prior_table'] and cfg['space'].get('mode', 'grid') == 'charge':
            logger.warning("charge mode: the table prior is not used (the table holds almost no per-electrode charges)")
        elif cfg['prior_table']:
            d = table_trials(cfg['prior_table'], self.joints, cfg.get('prior_dates'))
            prior = prior_from_table(cfg, d['X'], d['Y'])
            logger.info(f"prior mean from {len(d['X'])} table trains ({sorted(set(d['date']))}), scale {cfg['optimizer']['prior_scale']}")
        self.search = TargetSearch(cfg, prior)
        self.mode = cfg['mode']
        self.rng = np.random.default_rng(int(cfg['optimizer']['seed']))
        self.q_mc, self.t_start = 0.0, None                 # cumulative charge delivered this session (mC); time of the first request
        self._seed(cfg.get('seed_runs') or [])
        self.script = json.loads(Path(cfg['script']).expanduser().read_text()) if self.mode == 'script' else None
        self.cmp = None
        if self.mode == 'compare':
            c = dict(arms=['gp', 'random'], budget=30, verify_repeats=5, verify_model=True, learn_fixed=True, fixed_stims=None, reference=None)
            c.update(cfg.get('compare') or {})
            self.cmp = dict(cfg=c, t=None, phase=None, queue=deque(), arm_trials={}, answers={}, since_ref=0)
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

    def _seed(self, runs):
        """Earlier runs' measured trials (bo_trials.jsonl) as GP data, e.g. the same day's map before a search. In charge
        mode only trials at the space's amplitude and frequency are used. With fatigue 'charge', each seeded trial's input is
        the charge delivered before it in those runs, and this run's count starts at their total."""
        sp, n, q_run = self.cfg['space'], 0, 0.0
        for run in runs:
            for line in (Path(run).expanduser() / 'bo_trials.jsonl').read_text().splitlines():
                x = json.loads(line)
                q_before, q_run = q_run, q_run + float(x.get('charge_mc') or 0.0)
                ch, r = x.get('change') or {}, x['request']
                d = x.get('delivered') or {}
                if x.get('status', 'ok') != 'ok' or any(ch.get(j) is None for j in self.joints):
                    continue
                el, pw = d.get('electrodes') or r['electrodes'], d.get('pulse_width') or r['pulse_width']
                fq, amp = d.get('frequency') or r['frequency'], d.get('amplitude') or r['amplitude']
                if self.search.charge_mode:             # only trials played the way the search plays a charge (frequency, amplitude rule)
                    pws_, amps_ = np.broadcast_to(np.atleast_1d(pw), (len(el),)), np.broadcast_to(np.atleast_1d(amp), (len(el),))
                    rule = [charge_setting(a * p * 1e-6, int(sp['amplitude']), int(sp.get('min_pulse_width', 44)))[0] for a, p in zip(amps_, pws_)]
                    if int(fq) != int(sp['frequency']) or any(abs(int(a) - r) > 100 for a, r in zip(amps_, rule)):
                        continue
                self.search.add(self.search.encode_stim(el, pw, fq, amp), [ch[j] for j in self.joints], q=q_before)
                n += 1
        if runs:
            if self.cfg['optimizer'].get('fatigue') == 'charge':
                self.q_mc = q_run
            logger.info(f"seeded the GPs with {n} trials from {len(runs)} earlier run(s); {q_run:.1f} mC delivered in them")

    def _set_target(self, m):
        pass                                        # the targets are the config's list; BRAND's task target is not used here

    def _maybe_refit(self):
        pass                                        # the hyper-parameters are fixed while running (fitted beforehand)

    def _take_joints(self):
        """The output angles of the newest frames, from the processor's 3D keypoints."""
        for msg in self._drain('joints_in'):
            if not isinstance(msg, dict) or msg.get('points_3d') is None:
                self.no3d = getattr(self, 'no3d', 0) + 1           # 11-12: a placeholder calibration path left the BO silent
                if self.no3d % 600 == 1:
                    logger.warning(f"{self.no3d} frames without 3D hand points: nothing is requested until 3D arrives "
                                   f"(check calibration_toml / triangulate in the graph)")
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

    def _write_status(self, now):
        """<run folder>/bo_status.json for the GUI panel (actors/front_end_3d.py): where the experiment is, the last request,
        the miss per trial and the predicted miss over pulse width x frequency for the last request's electrodes and current."""
        s, base = self.search, self._resting(now)
        t = s.current_target()
        wanted = None if t is None or (s.posture and base is None) else s.wanted_change(t, base)
        done = self.trials[-1] if self.trials else None
        cur = self.trial or done
        miss = [float(np.linalg.norm(np.array([k['change'][j] for j in self.joints]) - s.wanted_change(k['target_index'], k['resting'])))
                for k in self.trials if k.get('change') and None not in k['change'].values() and k.get('target_index') is not None
                and k['target_index'] < len(s.targets) and k.get('arm') != 'reference' and (k.get('resting') is not None or not s.posture)]
        heat = None
        if cur is not None and wanted is not None and not s.charge_mode:          # charge mode has no width x rate slice
            r = cur['request']
            heat = dict(pw=s.pws, fq=s.fqs, electrodes=r['electrodes'], amplitude=r['amplitude'], pulse_width=r['pulse_width'],
                        frequency=r['frequency'], miss=s.miss_slice(r['electrodes'], r['amplitude'], wanted).tolist())
        st = dict(t=now, settings=self.settings, state=self.state, trial=self.n_trials, outputs=self.joints, posture=s.posture,
                  target=None if t is None else s.names[t], target_pose=None if t is None else s.targets[t].tolist(),
                  resting=None if base is None else base.tolist(), wanted=None if wanted is None else wanted.tolist(),
                  request=None if cur is None else cur['request'], mode=None if cur is None else cur['acquisition'].get('mode'),
                  last_change=None if done is None else done.get('change'), last_status=None if done is None else done.get('status'),
                  charge_mc=self.q_mc, booked=len(s.dictionary), n_targets=len(s.targets), miss=miss[-60:], heat=heat)
        tmp = self.out_folder / 'bo_status.json.tmp'
        tmp.write_text(json.dumps(st, default=float))
        tmp.replace(self.out_folder / 'bo_status.json')           # atomic: the GUI never reads half a file

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
        if self.mode == 'script':
            return self._request_script(now, base)
        if self.mode == 'compare':
            return self._request_compare(now, base)
        i, why = self.search.choose(base)
        if i is None and why.get('mode') == 'no eligible candidate':
            self.search.give_up()
            logger.info(f"no eligible candidate left: {self.search.dictionary[-1] if self.search.dictionary else ''}")
            i, why = self.search.choose(base)
        if i is None:
            return
        self._send_stim(now, *self.search.grid[i], why, base=base, candidate=i)

    def _send_stim(self, now, electrodes, pulse_width, frequency, amplitude, why, base=None, candidate=None, learn=True, duration=None, **extra):
        """One stim request. pulse_width / amplitude: one value, or one per electrode (BRAND's vision node takes both).
        learn: whether the measured change becomes GP data. extra: logged with the trial (arm, phase, label, ...)."""
        per = lambda v: [int(x) for x in v] if isinstance(v, (list, tuple, np.ndarray)) else int(v)
        self.req_id += 1
        req = dict(type='stim_request', id=self.req_id, electrodes=[int(e) for e in electrodes], amplitude=per(amplitude), pulse_width=per(pulse_width),
                   frequency=int(frequency), duration=float(duration or self.T['stim_duration']), t_sent=now)
        target = extra.pop('target_index', self.search.current_target() if self.mode != 'script' else None)
        self.trial = dict(id=self.req_id, request=req, t_sent=now, acquisition=why, candidate=candidate, q=self.search.q_now,
                          resting=None if base is None else base.tolist(), target_index=target, learn=bool(learn), **extra)
        self.q_out.put(req)
        self.state = 'waiting_on'
        logger.info(f"trial {self.n_trials + 1}: request {self.req_id} electrodes {req['electrodes']} {req['amplitude']} uA / "
                    f"{req['pulse_width']} us / {req['frequency']} Hz ({why.get('name') or why.get('label')}, {why['mode']}{', ' + extra['arm'] if extra.get('arm') else ''})")
        self._write_status(now)

    # -- mode 'script': a fixed list of stimulations, played in order (scripts/monday.py writes them)
    def _request_script(self, now, base):
        """Rows: {electrodes, pulse_width, amplitude, frequency, duration?, label, learn?} or {pause: seconds, label}."""
        while self.script and 'pause' in self.script[0]:
            row = self.script.pop(0)
            self.next_allowed = now + float(row['pause'])
            logger.info(f"script pause {row['pause']} s ({row.get('label', '')}); {len(self.script)} rows left")
            return
        if not self.script:
            if not getattr(self, '_script_done', False):
                self._script_done = True
                logger.info(f"script finished after {self.n_trials} trials, {self.q_mc:.1f} mC")
            return
        row = self.script.pop(0)
        why = dict(mode='script', label=row.get('label', ''), block=row.get('block', ''))
        self._send_stim(now, row['electrodes'], row['pulse_width'], row.get('frequency', self.cfg['space'].get('frequency', 50)),
                        row.get('amplitude', self.cfg['space'].get('amplitude', 5000)), why, base=base, learn=row.get('learn', False),
                        duration=row.get('duration'), label=row.get('label', ''), block=row.get('block', ''))

    # -- mode 'compare': per target, search arms interleaved trial by trial, then every answer verified
    def _stim_of(self, i):
        c, pw, fq, cu = self.search.grid[i]
        return dict(electrodes=list(c), pulse_width=list(pw) if isinstance(pw, tuple) else pw, frequency=fq, amplitude=cu)

    def _compare_start(self, t):
        C, cc = self.cmp, self.cmp['cfg']
        self.search.active = t
        C.update(t=t, phase='search', arm_trials={a: [] for a in cc['arms']}, answers={})
        if cc['verify_model']:                      # what the model alone would answer before this target's search
            m = self.search.posterior()[0] if self.search.optims is not None else self.search.prior_mean
            C['answers']['model'] = dict(candidate=int(np.argmin(np.linalg.norm(m - self.search.targets[t], axis=1))))
        rounds = [list(self.rng.permutation(cc['arms'])) for _ in range(int(cc['budget']))]
        C['queue'] = deque(a for r in rounds for a in r)
        logger.info(f"compare: target {self.search.names[t]} {self.search.targets[t].tolist()}, arms {cc['arms']} x {cc['budget']}")

    def _compare_answers(self):
        """Each arm's answer: GP = the tried stim whose posterior mean is nearest the target (as booking does); random = the
        tried stim whose single measurement was nearest; then every answer is repeated verify_repeats times, shuffled."""
        C, cc, t = self.cmp, self.cmp['cfg'], self.cmp['t']
        tau = self.search.targets[t]
        for arm, rows in C['arm_trials'].items():
            ok = [r for r in rows if r['change'] is not None]
            if not ok or arm == 'fixed':
                continue
            if arm == 'gp':
                m = self.search.posterior()[0]
                C['answers']['gp'] = dict(candidate=min({r['candidate'] for r in ok}, key=lambda i: float(np.linalg.norm(m[i] - tau))))
            elif arm == 'random':
                C['answers']['random'] = dict(candidate=min(ok, key=lambda r: float(np.linalg.norm(np.array(r['change']) - tau)))['candidate'])
        for a in C['answers'].values():
            a['verify'] = []
        C['queue'] = deque(self.rng.permutation([a for a in C['answers'] for _ in range(int(cc['verify_repeats']))]).tolist())
        C['phase'] = 'verify'

    def _compare_book(self):
        C, t = self.cmp, self.cmp['t']
        tau = self.search.targets[t]
        m, sd = self.search.posterior() if self.search.optims is not None else (self.search.prior_mean, None)
        miss = lambda ch: float(np.linalg.norm(np.array(ch) - tau))
        answers = {}
        for name, a in C['answers'].items():
            v = np.array([ch for ch in a['verify'] if ch is not None])
            answers[name] = dict(**self._stim_of(a['candidate']), predicted=m[a['candidate']].tolist(),
                                 verified_n=len(v), verified_mean=v.mean(0).tolist() if len(v) else None, verified_sd=v.std(0).tolist() if len(v) > 1 else None,
                                 verified_distance_deg=miss(v.mean(0)) if len(v) else None,
                                 verified_single_distance_deg=[miss(x) for x in v])
        entry = dict(target=tau.tolist(), name=self.search.names[t], outputs=self.joints, kind='compare', answers=answers,
                     arm_miss={arm: [None if r['change'] is None else miss(r['change']) for r in rows] for arm, rows in C['arm_trials'].items()})
        self.search.dictionary.append(entry)
        (self.out_folder / 'bo_dictionary.json').write_text(json.dumps(self.search.dictionary, indent=1))
        logger.info(f"compare booked {entry['name']}: " + ', '.join(f"{k} {v['verified_distance_deg'] and round(v['verified_distance_deg'], 1)} deg" for k, v in answers.items()))

    def _request_compare(self, now, base):
        C, cc = self.cmp, self.cmp['cfg']
        if C['t'] is None:
            self._compare_start(0)
        if C['phase'] == 'done':
            return
        if not C['queue'] and C['phase'] == 'search':      # search over: each arm's answer, then their verification
            self._compare_answers()
        if not C['queue'] and C['phase'] == 'verify':      # verification over: book, then the next target
            self._compare_book()
            if C['t'] + 1 >= len(self.search.targets):
                C['phase'] = 'done'
                logger.info(f"compare finished after {self.n_trials} trials, {self.q_mc:.1f} mC")
                return
            self._compare_start(C['t'] + 1)
        t, ref = C['t'], cc.get('reference')
        if ref and C['since_ref'] >= int(ref['every']):
            C['since_ref'] = 0
            sp = self.cfg['space']
            return self._send_stim(now, ref['electrodes'], ref['pulse_width'], ref.get('frequency', sp.get('frequency', 50)), ref.get('amplitude', sp.get('amplitude', 5000)),
                                   dict(mode='reference', label='reference'), base=base, learn=False, arm='reference', phase=C['phase'], target_index=t)
        C['since_ref'] += 1
        arm = C['queue'].popleft()
        name = self.search.names[t]
        if C['phase'] == 'verify':
            i = C['answers'][arm]['candidate']
            return self._send_stim(now, *self.search.grid[i], dict(mode='verify', name=name, answer=arm), base=base, candidate=i,
                                   learn=(arm == 'gp'), arm=arm, phase='verify', target_index=t)
        if arm == 'gp':
            i, why = self.search.choose(base)
            return self._send_stim(now, *self.search.grid[i], why, base=base, candidate=i, learn=True, arm='gp', phase='search', target_index=t)
        if arm == 'random':
            i = int(self.rng.integers(len(self.search.grid)))
            return self._send_stim(now, *self.search.grid[i], dict(mode='random', name=name), base=base, candidate=i, learn=False, arm='random', phase='search', target_index=t)
        if arm == 'fixed':
            f = cc['fixed_stims'][t]
            sp = self.cfg['space']
            return self._send_stim(now, f['electrodes'], f['pulse_width'], f.get('frequency', sp.get('frequency', 50)), f.get('amplitude', sp.get('amplitude', 5000)),
                                   dict(mode='fixed', name=name), base=base, learn=bool(cc['learn_fixed']), arm='fixed', phase='search', target_index=t)
        raise ValueError(f"unknown compare arm {arm}")

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
        amps, pws = np.broadcast_to(np.asarray(cu, float), (len(el),)), np.broadcast_to(np.asarray(pw, float), (len(el),))
        tr['charge_mc'] = float(sum(train_charge_mc(1, p, fq, a, dur) for a, p in zip(amps, pws)))
        if tr.get('learn', True):
            measured = self.search.add(self.search.encode_stim(el, pw, fq, cu), change, q=tr['q'], baseline=base)
        else:
            measured = bool(np.isfinite(change).all())
        self.q_mc += tr['charge_mc']                # the muscles got it whether or not the angles were measured
        done = None
        if self.mode == 'compare' and tr.get('arm') not in (None, 'reference'):
            C, ch = self.cmp, (change.tolist() if measured else None)
            (C['answers'][tr['arm']]['verify'] if tr.get('phase') == 'verify' else C['arm_trials'][tr['arm']]).append(
                ch if tr.get('phase') == 'verify' else dict(candidate=tr['candidate'], change=ch))
        elif self.mode not in ('script', 'compare') and measured:
            done = self.search.finish_trial(tr['candidate'], change)
        if self.search.optims is not None:
            t0 = time.time()                        # re-fit the GPs at the new "now" during the rest, not before the next request
            self.search.set_fatigue(self._fatigue_now(now)); self.search.posterior()
            tr['refit_s'] = time.time() - t0
        if done:
            logger.info(f"target finished ({done}): {self.search.dictionary[-1]}")
            (self.out_folder / 'bo_dictionary.json').write_text(json.dumps(self.search.dictionary, indent=1))
        self._close('ok' if measured else 'angles not measured', now)
        self._write_status(now)

    def stop(self):
        logger.info(f"BayesOptTargets stopping after {self.n_trials} trials, {self.q_mc:.1f} mC; {len(self.search.dictionary)} dictionary entries")
        (self.out_folder / 'bo_dictionary.json').write_text(json.dumps(self.search.dictionary, indent=1))
        s = self.search
        if s.obs_x:
            np.savez_compressed(self.out_folder / 'bo_gp_data.npz', outputs=np.array(s.outputs), X=np.array(s.obs_x), Y=np.array(s.obs_y),
                                prior=np.array(s.obs_prior), resting=np.array(s.obs_base))

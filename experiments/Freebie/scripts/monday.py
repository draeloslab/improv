"""Monday 2026-11-16, War Machine's last session: stim lists, settings, graphs, and the steps between blocks.
Procedure: Markdown/MONDAY_RUN_SHEET.md. Settings: config.yaml block `bo_monday`, overridden per block by
config/monday/b<N>.yaml (the graphs graphs/monday_b<N>.yaml pass it to BayesOptTargets as settings_file).

    python scripts/monday.py blocks                          # before: block 1-3 stim lists + settings (needs the 11-12 run logs)
    python scripts/monday.py graphs --calibration <toml>     # before / on the day: graphs/monday_b1..b6.yaml
    python scripts/monday.py check <run folder>              # after any block: trials, measured, charge, fatigue drift
    python scripts/monday.py targets <map run> [<run> ...]   # after block 3: refit on the map, targets, block 4 settings
    python scripts/monday.py after4 <block 4 run>            # after block 4: block 5 sequence + block 6 fatigue settings
    python scripts/monday.py simulate                        # offline: the head-to-head on a surrogate (budget, k)

Blocks: 1 anchor + re-test of 11-12's near hits, 2 frequency + charge-equivalence checks, 3 same-day map, 4 GP-BO vs
random search vs the map's model on 3 targets with verification, 5 posture sequence, 6 fatigue (fixed vs adaptive).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

FES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FES))
from actors.target_search import ELECTRODES, TargetSearch, charge_setting, encode_charge, fit_hyperparameters, merged  # noqa: E402

OUT = FES / 'config' / 'monday'
REFERENCE = dict(electrodes=[4, 16], charges=[0.75, 0.75], every=15)       # ECRB_1 (wrist) + EDC_4 (fingers), one drift probe
MAP_LEVELS = [0.06, 0.1, 0.16, 0.25, 0.4, 0.6, 0.85, 1.25]                  # single-electrode recruitment, uC per phase
NEAR_HITS = [('20261112-1300', 0, '11-12 A open_hand'), ('20261112-1300', 1, '11-12 A ulnar_wave'),
             ('20261112-1307', 0, '11-12 B extension'), ('20261112-1313', 0, '11-12 C1 neutral')]
GOALS = {'wrist_up': lambda y: y[0] - abs(y[1]) - 0.5 * abs(y[2]),           # outputs: WRIST_EXT, WRIST_DEV, FINGER_EXT
         'open_hand': lambda y: y[2] - abs(y[1]) - 0.5 * abs(y[0] - 5.0),
         'ulnar': lambda y: y[1] - 0.5 * abs(y[2]) - 0.5 * abs(y[0])}


def config():
    c = yaml.safe_load((FES / 'config' / 'config.yaml').read_text())
    return merged(c['bayes_opt_targets'], c['bo_monday'])


def row(electrodes, charges, label, block, frequency=50, setting=None, duration=1.0):
    """One script row. charges in uC per electrode, played as charge_setting() unless `setting` gives (amplitude, width)."""
    s = [setting or charge_setting(q) for q in charges]
    return dict(electrodes=[int(e) for e in electrodes], amplitude=[a for a, _ in s], pulse_width=[w for _, w in s],
                frequency=int(frequency), duration=float(duration), label=label, block=block)


def row_charge_mc(r):
    return sum(a * w for a, w in zip(r['amplitude'], r['pulse_width'])) * 1e-6 * round(r['frequency'] * r['duration']) * 1e-3


def shuffled(rows, rng, key='label'):
    """Random order with no row label twice in a row (fatigue: the same muscle is not hit back to back)."""
    for _ in range(1000):
        order = [rows[i] for i in rng.permutation(len(rows))]
        if all(a[key] != b[key] for a, b in zip(order, order[1:])):
            return order
    return order


def with_references(rows, block, every=REFERENCE['every'], pause_every=None, pause_s=60):
    out, n = [], 0
    for r in rows:
        if n and n % every == 0:
            out.append(row(REFERENCE['electrodes'], REFERENCE['charges'], 'reference', block))
        out.append(r); n += 1
        if pause_every and n % pause_every == 0 and n < len(rows):
            out.append(dict(pause=pause_s, label=f'rest after {n} trials', block=block))
    return out


def near_hits():
    """The 11-12 trial nearest its wanted change for each NEAR_HITS target (C1: after the release, trial >= 25)."""
    out = []
    for run, t, label in NEAR_HITS:
        best = None
        for k, line in enumerate((Path('~/predictions/20261112').expanduser() / run / 'bo_trials.jsonl').read_text().splitlines()):
            x = json.loads(line)
            ch, acq = x.get('change') or {}, x.get('acquisition') or {}
            if x.get('target_index') != t or not ch or None in ch.values() or 'wanted' not in acq or (run.endswith('1313') and k < 25):
                continue
            miss = float(np.linalg.norm(np.array(list(ch.values())) - np.array(acq['wanted'])))
            if best is None or miss < best[0]:
                best = (miss, x['delivered'], k)
        miss, d, k = best
        amp, pw = np.broadcast_to(d['amplitude'], len(d['electrodes'])), np.broadcast_to(d['pulse_width'], len(d['electrodes']))
        out.append(dict(electrodes=d['electrodes'], amplitude=[int(a) for a in amp], pulse_width=[int(w) for w in pw],
                        frequency=int(d['frequency']), duration=1.0, label=label, block='b1'))
        print(f"  near hit {label}: trial {k} miss {miss:.1f} deg -> electrodes {d['electrodes']} {d['amplitude']} uA / {d['pulse_width']} us / {d['frequency']} Hz")
    return out


def map_rows(rng, n_combos=216, n_repeats=30, pool=20000, warp=0.05):
    """Single-electrode recruitment (8 electrodes x MAP_LEVELS) + space-filling 2-4 electrode combinations with an
    independent charge each (greedy max-min distance in the GP's input space) + n_repeats of them played twice."""
    levels = config()['space']['charges']
    singles = [row([e], [q], f'e{e} {q:g}uC', 'b3') for e in ELECTRODES for q in MAP_LEVELS]
    cand = []
    for _ in range(pool):
        els = sorted(rng.choice(ELECTRODES, int(rng.integers(2, 5)), replace=False).tolist())
        cand.append((els, [float(rng.choice(levels)) for _ in els]))
    enc = lambda els, qs: encode_charge(els, [charge_setting(q)[1] for q in qs], [charge_setting(q)[0] for q in qs], warp)[:8]
    C = np.array([enc(*c) for c in cand]); S = np.array([enc(r['electrodes'], [a * w * 1e-6 for a, w in zip(r['amplitude'], r['pulse_width'])]) for r in singles])
    dmin = np.min(np.linalg.norm(C[:, None] - S[None], axis=2), axis=1)
    pick = []
    for _ in range(n_combos):
        i = int(np.argmax(dmin)); pick.append(i)
        dmin = np.minimum(dmin, np.linalg.norm(C - C[i], axis=1)); dmin[i] = -1
    combos = [row(cand[i][0], cand[i][1], f'combo {k}', 'b3') for k, i in enumerate(pick)]
    repeats = [dict(r, label=r['label'] + ' (repeat)') for r in (combos[i] for i in rng.choice(len(combos), n_repeats, replace=False))]
    return singles + combos + repeats


def write_block(name, rows, settings):
    (OUT / f'{name}.json').write_text(json.dumps(rows, indent=0))
    (OUT / f'{name.split("_")[0]}.yaml').write_text(yaml.safe_dump(dict(settings, script=str(OUT / f'{name}.json')), sort_keys=False))
    stims = [r for r in rows if 'pause' not in r]
    secs = 3.0 * len(stims) + sum(r.get('pause', 0) for r in rows)
    print(f"{name}: {len(stims)} trains ({sum(r['label'] == 'reference' for r in stims)} references), ~{secs / 60:.1f} min, ~{sum(map(row_charge_mc, stims)):.1f} mC")
    return len(stims), sum(map(row_charge_mc, stims))


def cmd_blocks(a):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(16)
    b1 = [row([e], [0.75], f'anchor e{e}', 'b1') for e in ELECTRODES for _ in range(2)] + [dict(r) for r in near_hits() for _ in range(5)]
    b2 = [row([e], [q], f'freq e{e} {q:g}uC {f}Hz', 'b2', frequency=f) for e in (4, 6, 16) for q in (0.6, 1.0) for f in (30, 40, 50) for _ in range(2)]
    b2 += [row([e], [q], f'equiv e{e} {q:g}uC {s[0]}uA x {s[1]}us', 'b2', setting=s) for e in (4, 16)
           for q, ss in ((0.75, [(3000, 250), (5000, 150), (10000, 75)]), (0.2, [(4500, 44), (2000, 100), (1000, 200)])) for s in ss for _ in range(2)]
    b3 = map_rows(rng)
    total = [write_block('b1_anchor', shuffled(b1, rng), dict(mode='script')),
             write_block('b2_checks', shuffled(b2, rng), dict(mode='script')),
             write_block('b3_map', with_references(shuffled(b3, rng), 'b3', pause_every=100), dict(mode='script'))]
    print(f"blocks 1-3: {sum(n for n, _ in total)} trains, ~{sum(q for _, q in total):.0f} mC -> {OUT}")


def cmd_graphs(a):
    src = (FES / 'graphs' / 'bo_targets_live_A.yaml').read_text().splitlines()
    for n in range(1, 7):
        lines = []
        for line in src:
            if line.strip().startswith('BayesOptTargets:'):
                line = (f"  BayesOptTargets: {{package: actors.bayes_opt_targets, class: BayesOptTargets, method: spawn, settings: bo_monday, "
                        f"settings_file: '{OUT / f'b{n}.yaml'}'}}")
            elif 'calibration_toml:' in line:
                line = f"      calibration_toml: '{Path(a.calibration).expanduser().resolve()}'"
            elif line.startswith('# Live GP-BO towards hand postures'):
                line = f"# Monday 2026-11-16, block {n} (written by scripts/monday.py graphs; procedure: Markdown/MONDAY_RUN_SHEET.md)."
            lines.append(line)
        (FES / 'graphs' / f'monday_b{n}.yaml').write_text('\n'.join(lines) + '\n')
    print(f"graphs/monday_b1..b6.yaml written with calibration {a.calibration}")


def run_trials(run, outputs):
    """(X, Y, rows) of a run's measured trials played the way the charge search plays a charge (50 Hz, the amplitude rule)."""
    cfg = config(); sp = cfg['space']
    X, Y, rows = [], [], []
    for line in (Path(run).expanduser() / 'bo_trials.jsonl').read_text().splitlines():
        x = json.loads(line); ch = x.get('change') or {}; d = x.get('delivered') or x['request']
        if x.get('status', 'ok') != 'ok' or any(ch.get(j) is None for j in outputs):
            continue
        el = d['electrodes']; amp, pw = np.broadcast_to(d['amplitude'], len(el)), np.broadcast_to(d['pulse_width'], len(el))
        if int(d['frequency']) != int(sp['frequency']) or any(abs(int(p) - charge_setting(int(p) * int(w) * 1e-6)[0]) > 100 for p, w in zip(amp, pw)):
            continue
        X.append(encode_charge(el, pw, amp, sp['charge_warp'])); Y.append([ch[j] for j in outputs]); rows.append(x)
    return np.array(X), np.array(Y), rows


def cmd_targets(a):
    cfg = config(); outs = cfg['outputs']
    parts = [run_trials(r, outs) for r in a.runs]
    X, Y = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    labels = [((x.get('acquisition') or {}).get('label') or '') for p in parts for x in p[2]]
    print(f"{len(X)} usable trials from {len(a.runs)} run(s)")
    o = cfg['optimizer']
    h = fit_hyperparameters(X, Y, ard=False)
    inside = (0.55 < h['electrode_length'] < 3.9 and all(0.6 < v < 75 for v in h['prior_sd']) and all(0.35 < v < 28 for v in h['noise_sd']))
    ok = len(X) >= 100 and inside                                      # a day's map, and no parameter stuck at a bound of the fit
    noise = [max(v, 0.8 * c) for v, c in zip(h['noise_sd'], o['noise_sd'])]   # never more confident than 0.8 x the table's noise (live overconfidence)
    hp = dict(electrode_length=round(h['electrode_length'], 3), prior_sd=[round(v, 2) for v in h['prior_sd']], noise_sd=[round(v, 2) for v in noise]) if ok else \
        dict(electrode_length=o['electrode_length'], prior_sd=o['prior_sd'], noise_sd=o['noise_sd'])
    print(f"refit on the day ({len(X)} trials): length {h['electrode_length']:.2f}, prior_sd {np.round(h['prior_sd'], 1).tolist()}, noise_sd {np.round(h['noise_sd'], 1).tolist()} "
          f"-> {'USED (noise floor 0.8 x config)' if ok else 'NOT used (under 100 trials or a parameter at a bound): config values kept'}")
    Yt = Y[np.array([lab != 'reference' for lab in labels])]
    targets, names = [], []
    for name, score in GOALS.items():
        order = np.argsort([score(y) for y in Yt])[::-1]
        start = len(order) - 1 - int(0.9 * (len(order) - 1))           # the 90th-percentile trial for this goal, not the extreme
        i = next((int(j) for j in order[start:] if all(np.linalg.norm(Yt[j] - np.array(t)) >= 10 for t in targets)), int(order[start]))
        tau = np.round(Yt[i]).astype(int).tolist()
        near = int((np.linalg.norm(Yt - Yt[i], axis=1) < 10).sum())
        targets.append(tau); names.append(name)
        print(f"  target {name}: {tau} (measured on the map; {near} map trials within 10 deg)" + ("   ** under 10 deg from rest: check the map **" if np.linalg.norm(tau) < 10 else ""))
    b4 = dict(mode='compare', seed_runs=[str(Path(r).expanduser().resolve()) for r in a.runs], targets=targets, target_names=names,
              optimizer=dict(hp, k=float(a.k)), compare=dict(arms=['gp', 'random'], budget=int(a.budget), verify_repeats=5, verify_model=True,
              reference=dict(electrodes=REFERENCE['electrodes'], pulse_width=[charge_setting(q)[1] for q in REFERENCE['charges']],
                             amplitude=[charge_setting(q)[0] for q in REFERENCE['charges']], every=REFERENCE['every'])))
    (OUT / 'b4.yaml').write_text(yaml.safe_dump(b4, sort_keys=False))
    n = len(targets) * (2 * int(a.budget) + 15); n += n // REFERENCE['every']
    print(f"block 4: {len(targets)} targets x (2 x {a.budget} + 3 answers x 5) + references = ~{n} trains, ~{n * 3 / 60:.0f} min -> {OUT / 'b4.yaml'}")


def cmd_after4(a):
    run = Path(a.run).expanduser().resolve()
    entries = [e for e in json.loads((run / 'bo_dictionary.json').read_text()) if e.get('kind') == 'compare']
    best = []
    for e in entries:
        ok = {k: v for k, v in e['answers'].items() if v['verified_n'] >= 3}
        k = min(ok, key=lambda k: ok[k]['verified_distance_deg'])
        v = ok[k]; best.append((e['name'], v))
        print(f"  {e['name']} {e['target']}: " + ', '.join(f"{n} {w['verified_distance_deg'] and round(w['verified_distance_deg'], 1)} deg (n={w['verified_n']})" for n, w in e['answers'].items())
              + f" -> sequence uses {k}")
    seq = [dict(electrodes=v['electrodes'], amplitude=np.broadcast_to(v['amplitude'], len(v['electrodes'])).tolist(), pulse_width=np.broadcast_to(v['pulse_width'], len(v['electrodes'])).tolist(),
                frequency=int(v['frequency']), duration=1.0, label=f'sequence {name}', block='b5') for _ in range(10) for name, v in best]
    write_block('b5_sequence', seq, dict(mode='script'))
    session = sorted(str(p.parent) for p in run.parent.glob('*/bo_settings.json') if json.loads(p.read_text()).get('settings') == 'bo_monday' and p.parent.name <= run.name)
    b4 = json.loads((run / 'bo_settings.json').read_text())['cfg']     # what block 4 actually ran with
    name, v = best[0]                                                  # fatigue on the first target (wrist_up)
    b6 = dict(mode='compare', seed_runs=session, targets=[b4['targets'][0]], target_names=[name + '_fatigue'],
              optimizer=dict({k: b4['optimizer'][k] for k in ('electrode_length', 'prior_sd', 'noise_sd', 'k')}, fatigue='charge', fatigue_length=20.0),
              compare=dict(arms=['fixed', 'gp'], budget=75, verify_repeats=0, verify_model=False, learn_fixed=True, reference=None,
                           fixed_stims=[dict(electrodes=v['electrodes'], amplitude=v['amplitude'], pulse_width=v['pulse_width'])]),
              timing=dict(stim_duration=1.0, rest_duration=1.0))
    (OUT / 'b6.yaml').write_text(yaml.safe_dump(b6, sort_keys=False))
    print(f"block 6: fixed {v['electrodes']} {v['amplitude']} uA / {v['pulse_width']} us vs the charge-aware GP, 2 x 75 trains (~5 min); seeded with {len(session)} runs -> {OUT / 'b6.yaml'}")


def cmd_check(a):
    run = Path(a.run).expanduser()
    tr = [json.loads(l) for l in (run / 'bo_trials.jsonl').read_text().splitlines()]
    st = json.loads((run / 'bo_settings.json').read_text()); outs = st['cfg']['outputs']
    meas = [x for x in tr if x.get('change') and None not in x['change'].values()]
    span = (tr[-1]['t_sent'] - tr[0]['t_sent']) / 60 if tr else 0
    print(f"{run.name}: {len(tr)} trials in {span:.1f} min, {len(meas)} measured ({100 * len(meas) / max(len(tr), 1):.0f} %), "
          f"statuses {dict(zip(*np.unique([x.get('status') for x in tr], return_counts=True)))}, {sum(x.get('charge_mc') or 0 for x in tr):.1f} mC")
    ref = [x for x in meas if x.get('arm') == 'reference' or x.get('label') == 'reference']
    if len(ref) >= 4:
        R = np.array([[x['change'][j] for j in outs] for x in ref]); k = max(len(R) // 4, 2)
        first, last = R[:k].mean(0), R[-k:].mean(0)
        print(f"  reference ({len(R)}): first {np.round(first, 1).tolist()} -> last {np.round(last, 1).tolist()} deg ({outs})")
        drop = [100 * (1 - l / f) for f, l in zip(first, last) if abs(f) >= 8]
        if drop:
            print(f"  response drop on the outputs it moves: {np.round(drop, 0).tolist()} %" + ("   ** OVER 30 %: fatigue, move on **" if max(drop) > 30 else ""))
    d = run / 'bo_dictionary.json'
    for e in (json.loads(d.read_text()) if d.exists() else []):
        if e.get('kind') == 'compare':
            print(f"  {e['name']} {e['target']}: " + ', '.join(f"{n} {v['verified_distance_deg'] and round(v['verified_distance_deg'], 1)} deg" for n, v in e['answers'].items()))


def cmd_simulate(a):
    """Surrogate: the GP posterior mean fitted to one table day (charge encoding) is the 'true' response; noise from the
    fit. The block-3 map is simulated, targets chosen as on the day, then GP-BO (k values), random search and the map's
    model are compared on the true error of their answers after 10 / 20 / 30 trials. Optimistic: the truth is as smooth
    as the model assumes, and the table has almost no trains below 0.2 uC."""
    import pandas as pd
    from actors.target_search import derived
    cfg = config(); outs, sp, o = cfg['outputs'], cfg['space'], cfg['optimizer']
    t = pd.read_csv(Path('~/dlc_projects/FESHand21SingleHand-jake-2026-10-21/analysis/stim_table_it9/stim_table.csv').expanduser(), low_memory=False)
    u = t[t.timing_ok & t.timing.isin(['single', 'synchronous']) & (t.duration_s >= 0.9) & t.kind.isin(['plain']) & (t.date == a.day)]
    dd = derived({c[2:]: u[c].to_numpy(float) for c in u.columns if c.startswith('d_') and c != 'd_WRIST_EXT'} | {'WRIST_EXT': u.d_WRIST_EXT.to_numpy(float)})
    Y0 = np.stack([dd[k] for k in outs], 1); X0, Yt = [], []
    for n, (_, r) in enumerate(u.iterrows()):
        els = [int(e) for e in r.electrodes.split('-')]; par = np.array([[r[f'e{e}_w1'], r[f'e{e}_rate'], r[f'e{e}_a1']] for e in els], float)
        if np.isfinite(par).all() and par[0, 1] >= 30 and np.isfinite(Y0[n]).all():
            X0.append(encode_charge(els, par[:, 0], par[:, 2], sp['charge_warp'])); Yt.append(Y0[n])
    X0, Yt = np.array(X0), np.array(Yt)
    scfg = merged(cfg, dict(space=dict(n_random=a.n_random)))
    truth = TargetSearch(scfg); truth.obs_x, truth.obs_y, truth.obs_prior = list(X0), list(Yt), [np.zeros(3)] * len(X0); truth._initialize()
    F = truth.posterior()[0]                                            # true mean response of every candidate
    def f_at(X):
        ts = TargetSearch.__new__(TargetSearch); ts.__dict__.update(truth.__dict__); ts.X = np.asarray(X); ts.prior_mean = np.zeros((len(X), 3)); ts._ks_grid = []; ts.q_fit = None
        return ts.posterior()[0]
    noise = np.asarray(o['noise_sd'], float)
    rng0 = np.random.default_rng(16); maprows = map_rows(rng0)
    Xmap = np.array([encode_charge(r['electrodes'], r['pulse_width'], r['amplitude'], sp['charge_warp']) for r in maprows]); Fmap = f_at(Xmap)
    print(f"surrogate from {a.day}: {len(X0)} trains; {len(F)} candidates; best reachable spread of outputs {np.round(F.min(0), 0).tolist()} .. {np.round(F.max(0), 0).tolist()}")
    res = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        Ymap = Fmap + rng.normal(0, noise, Fmap.shape)
        for name, score in GOALS.items():
            s = np.array([score(y) for y in Ymap]); tau = np.round(Ymap[int(np.argsort(s)[int(0.9 * (len(s) - 1))])])
            floor = float(np.min(np.linalg.norm(F - tau, axis=1)))
            for k in a.k:
                ts = TargetSearch(merged(scfg, dict(targets=[tau.tolist()], optimizer=dict(k=k))))
                for x, y in zip(Xmap, Ymap): ts.add(x, y)
                m = ts.posterior()[0]; i_model = int(np.argmin(np.linalg.norm(m - tau, axis=1)))
                tried = []
                for n in range(1, a.budget + 1):
                    i, _ = ts.choose(); y = F[i] + rng.normal(0, noise); ts.add(ts.X[i], y); tried.append(i)
                    if n in (10, 20, 30, a.budget):
                        m = ts.posterior()[0]; ans = min(set(tried), key=lambda j: float(np.linalg.norm(m[j] - tau)))
                        res.append(dict(seed=seed, target=name, arm=f'gp k={k}', n=n, err=float(np.linalg.norm(F[ans] - tau)), floor=floor))
                res.append(dict(seed=seed, target=name, arm='model (map only)', n=0, err=float(np.linalg.norm(F[i_model] - tau)), floor=floor))
            ri = rng.integers(len(F), size=a.budget); ry = F[ri] + rng.normal(0, noise, (a.budget, 3))
            for n in (10, 20, 30, a.budget):
                b = int(np.argmin(np.linalg.norm(ry[:n] - tau, axis=1)))
                res.append(dict(seed=seed, target=name, arm='random', n=n, err=float(np.linalg.norm(F[ri[b]] - tau)), floor=floor))
        print(f"  seed {seed} done", flush=True)
    import pandas as pd
    R = pd.DataFrame(res).drop_duplicates()
    print('\ntrue distance of the answer to the target (deg), mean over seeds and targets [floor = best candidate]:')
    print(R.groupby(['arm', 'n']).err.mean().unstack().round(1).to_string())
    print('floor (best possible):', round(R.floor.mean(), 1), '| per target:', R.groupby('target').floor.mean().round(1).to_dict())
    print('\nper target, n = budget:'); print(R[R.n.isin([0, a.budget])].groupby(['target', 'arm']).err.mean().unstack().round(1).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('blocks')
    g = sub.add_parser('graphs'); g.add_argument('--calibration', required=True)
    c = sub.add_parser('check'); c.add_argument('run')
    t = sub.add_parser('targets'); t.add_argument('runs', nargs='+'); t.add_argument('--budget', type=int, default=30); t.add_argument('--k', type=float, default=0.5)
    f = sub.add_parser('after4'); f.add_argument('run')
    s = sub.add_parser('simulate'); s.add_argument('--day', default='2026-10-21'); s.add_argument('--seeds', type=int, default=4)
    s.add_argument('--budget', type=int, default=30); s.add_argument('--k', type=float, nargs='+', default=[0.3, 0.5, 1.0]); s.add_argument('--n-random', type=int, default=15000)
    a = ap.parse_args()
    {'blocks': cmd_blocks, 'graphs': cmd_graphs, 'check': cmd_check, 'targets': cmd_targets, 'after4': cmd_after4, 'simulate': cmd_simulate}[a.cmd](a)


if __name__ == '__main__':
    main()

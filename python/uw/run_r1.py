"""UW attack-paper R1 verdict execution (external review 2026-10-02).

Three arms (reviewer's minimum-revision items 1-3), all on the
CIFAR-10 ft-family frozen split (fit=ew9f 6, cal=p1aud+e2 12,
test=e3+e4 12, starts=e3):

  reanalysis : zero-training re-analysis -- (a) held-out (test-12)
               honest pass counts per alpha from the frozen scores;
               (b) per-start commit outcomes for the NO arms under a
               proxy-best selection rule (no true-audit feedback):
               the attacker submits the start whose final has the
               best O2 objective, judged once by the true audit;
               (c) symmetric-band and one-sided-cap honest rejection
               counts on test-12; (d) E_2 and absolute O2 margins.
  bias       : class-bias baseline -- add b >= 0 to the cat logit
               (final-layer bias) of each honest start, take the
               largest b that passes the TRUE audit, and report O2
               gain / F-accuracy / flips vs the start itself on the
               SAME O2 subset (reviewer IV.2).
  tdefense   : directly-aligned per-record score -- T(theta) =
               max_j (r_j - c_j)+ / sigma_j with per-record centers/
               scales frozen on the fit 6, conformal-calibrated on
               cal 12; honest pass on test 12; O2 attack under
               (T <= qT) alone and under (s <= qs AND T <= qT)
               (reviewer V.5).
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np

from uw.run_ey import (GATE, METRICS, conformal_q, gain_of,
                       per_record_margin, record_subsets)
from uw.run_ez import (CELLS, N_FT, apply_delta, make_context, obj_val,
                       run_attack, seed_all, setup_c10)

CACHE = sorted(glob.glob(
    'research_outputs/uw/m3-2*/metrics_cache.json'))[-1]
RUN3 = 'research_outputs/uw/ew3-20260925T113900'
ALPHAS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)


def load_audit():
    ens = json.load(open(CACHE))['ens_mm']
    fit = ens['ew9f']
    cal = [e for t in ('p1aud', 'e2') for e in ens[t]]
    test = [e for t in ('e3', 'e4') for e in ens[t]]
    c = {m: float(np.median([e[m] for e in fit])) for m in METRICS}
    sig = {m: float(np.median(np.abs(
        np.array([e[m] for e in fit]) - c[m]))) for m in METRICS}

    def sc(e):
        return max(abs(e[m] - c[m]) / max(sig[m], 1e-12)
                   for m in METRICS)

    q = {a: conformal_q([sc(e) for e in cal], a) for a in ALPHAS}
    return fit, cal, test, c, sig, sc, q


def run_reanalysis(cli):
    fit, cal, test, c, sig, sc, q = load_audit()
    out = {}

    # (a) honest pass counts: in-sample 24 (cal+test) vs held-out 12
    rows = {}
    for a in ALPHAS:
        ins = int(sum(sc(e) <= q[a] for e in cal + test))
        ho = int(sum(sc(e) <= q[a] for e in test))
        rows[a] = {'q': float(q[a]), 'in24': [ins, 24],
                   'heldout12': [ho, 12]}
    out['pass_counts'] = rows
    print('held-out pass per alpha:',
          {a: r['heldout12'] for a, r in rows.items()}, flush=True)

    # (b) per-start commit outcomes for oracle / NO arms (ey2 records)
    ey2 = sorted(glob.glob('research_outputs/uw/ey2-2*'))[-1]
    arms = {}
    for pref in ('orc_e3', 'orc_e4', 'no1_e3', 'no2_e3', 'rt_aud'):
        recs = []
        for p in sorted(glob.glob(os.path.join(
                ey2, '{}_O2_s*.json'.format(pref)))):
            r = json.load(open(p))
            r['start'] = os.path.basename(p)
            recs.append(r)
        if not recs:
            continue
        best_obj = max(r['best'] for r in recs)
        commit = [r for r in recs if r['best'] == best_obj][0]
        arms[pref] = {
            'per_start': [{'best': r['best'],
                           'est_pass': r.get('est_pass'),
                           'true_pass': r.get(
                               'true_pass', r.get('est_pass'))}
                          for r in recs],
            'proxy_commit_rule': 'submit the start with best final '
                                 'O2 objective (own info only)',
            'committed': {'best': commit['best'],
                          'true_pass': commit.get(
                              'true_pass', commit.get('est_pass'))},
            'pass_frac': float(np.mean(
                [r.get('true_pass', r.get('est_pass'))
                 for r in recs]))}
        print(pref, 'per-start pass',
              [r.get('true_pass', r.get('est_pass'))
               for r in recs], '| proxy-commit pass',
              arms[pref]['committed']['true_pass'], flush=True)
    out['commit'] = arms

    # (c) symmetric band / one-sided cap honest rejections on test-12
    fm_fit = [e['m6_fm'] for e in fit]
    fm_med = float(np.median(fm_fit))
    fm_b = float(max(abs(e['m6_fm'] - fm_med) for e in fit))
    fm_max = float(max(fm_fit))
    sym = int(sum(not (abs(e['m6_fm'] - fm_med) <= fm_b
                       and sc(e) <= q[0.20]) for e in test))
    cap = int(sum(not (e['m6_fm'] <= fm_max) for e in test))
    sym_only = int(sum(not (abs(e['m6_fm'] - fm_med) <= fm_b)
                       for e in test))
    out['defense_reject_test12'] = {
        'fm_med': fm_med, 'fm_b': fm_b, 'fm_max': fm_max,
        'sym_band_joint_rejects': sym, 'sym_band_alone_rejects': sym_only,
        'onecap_alone_rejects': cap}
    print('defense rejects on test-12: sym-joint', sym, 'sym-alone',
          sym_only, 'cap-alone', cap, flush=True)

    # (d) E_2 and absolute O2 margins per key unit (from verdict files)
    units = {}
    for pat, tag in (('ez2-c100-2*', 'c100-u1'),
                     ('ez7-c100b-2*', 'c100-u2'),
                     ('ez5-retrainfix-2*', 'retrainfix'),
                     ('ey-2026093*', 'ey-primary')):
        d = sorted(glob.glob('research_outputs/uw/' + pat
                             + '/verdict.json'))
        if not d:
            continue
        v = json.load(open(d[-1]))
        e = None
        for key in ('E_guard',):
            if key in v and 'O2' in v[key]:
                e = v[key]['O2']
        if e is None:
            continue
        units[tag] = {'E_O2': e.get('E'), 'std': e.get('pool_std'),
                      'theta0_O2': v.get('theta0_objs', {}).get('O2')}
    out['E_O2_units'] = units
    print('E_O2 units:', units, flush=True)

    with open(os.path.join(cli.outdir, 'reanalysis.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('OUT', cli.outdir, flush=True)


def run_bias(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    model, xF, yF = ctx['model'], ctx['xF'], ctx['yF']
    _, _, _, c, sig, sc, q = load_audit()
    q20 = q[0.20]

    sd0 = torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                     map_location=device)
    th_m = ctx['margins_of'](sd0)
    bot, top = record_subsets(th_m, k=10)
    th_o2 = obj_val(th_m, 'O2', bot, top)

    all_starts = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    # attacker's honest candidates that PASS the true audit (a
    # failing start cannot be repaired by adding class bias)
    passing = [p for p in all_starts
               if sc(ctx['eval_full'](torch.load(p,
                                                 map_location=device)))
               <= q20]
    n = 3 if not cli.smoke else min(1, len(passing))
    starts = passing[:n]
    grid = [0.5, 1.0, 2.0, 4.0, 8.0] if not cli.smoke else [1.0, 4.0]
    cat = 3
    out = {'grid': grid, 'starts': []}
    ck = os.path.join(cli.outdir, 'bias_sd')
    os.makedirs(ck, exist_ok=True)
    for p in starts[:n]:
        sd = torch.load(p, map_location=device)
        start_mm = ctx['eval_full'](sd)
        start_mg = ctx['margins_of'](sd)
        start_o2 = obj_val(start_mg, 'O2', bot, top)
        rec = {'start': os.path.basename(p),
               'start_o2': start_o2,
               'start_Facc': float((start_mg > 0).mean()),
               'b_pass': None, 'b_max_passing': None}
        best_b, best_rec = None, None
        for b in grid:
            sd2 = {k: v.clone() for k, v in sd.items()}
            sd2['fc.bias'][cat] += b
            mm = ctx['eval_full'](sd2)
            mg = ctx['margins_of'](sd2)
            o2 = obj_val(mg, 'O2', bot, top)
            passes = sc(mm) <= q20
            row = {'b': b, 's': float(sc(mm)), 'pass': bool(passes),
                   'o2': o2,
                   'F_acc': float((mg > 0).mean()),
                   'flips_mem_vs_start': int(((mg > 0) &
                                              ~(start_mg > 0)).sum()),
                   'flips_loss_vs_start': int(((start_mg > 0) &
                                               ~(mg > 0)).sum())}
            rec.setdefault('sweep', []).append(row)
            if passes:
                best_b, best_rec = b, row
        if best_rec is not None:
            med = start_o2  # gain vs start-anchored baseline
            rec['b_max_passing'] = best_b
            rec['o2_at_bmax'] = best_rec['o2']
            rec['gain_vs_theta0_delta'] = None  # filled below
        # E_2 for normalization: theta0 O2 - honest median O2
        o2s = [obj_val(ctx['margins_of'](torch.load(pp,
                                                    map_location=device)),
                       'O2', bot, top) for pp in all_starts]
        med = float(np.median(o2s))
        sel = max(o2s)
        E = th_o2 - med
        rec['E2'] = E
        if best_rec is not None:
            rec['gain_norm'] = float(gain_of(best_rec['o2'], sel, med,
                                             th_o2))
        out['starts'].append(rec)
        print(' bias', rec['start'], 'b_max', rec.get('b_max_passing'),
              'gain_norm', rec.get('gain_norm'), flush=True)

    # attacker-side usage: best passing b over the grid, gain vs
    # objective-matched selection from the same pool (honest O2 max)
    with open(os.path.join(cli.outdir, 'bias.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    gains = [r.get('gain_norm') for r in out['starts']
             if r.get('gain_norm') is not None]
    verdict = ('BIAS_REPLICATES' if gains and max(gains) >= GATE
               else 'BIAS_WEAK_OR_BLOCKED')
    out['verdict'] = verdict
    with open(os.path.join(cli.outdir, 'bias.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', verdict, flush=True)


def run_tdefense(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 2 if cli.smoke else 60
    n_starts = 1 if cli.smoke else 3
    ctx = setup_c10(device)
    model, xF, yF = ctx['model'], ctx['xF'], ctx['yF']
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]

    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-2*/ckpt/ft_k30_s*.pt'))
    cal_paths = (sorted(glob.glob(
        'research_outputs/uw/r3p1-2*/ckpt/ft_aud_s*.pt'))
        + sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/e2_ft_s*.pt')))
    test_paths = (sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
        + sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt')))
    starts = test_paths[:6]

    def mgs(paths):
        return [ctx['margins_of'](torch.load(p, map_location=device))
                for p in paths]

    fit_mg = mgs(fit_paths)
    R = np.stack(fit_mg)                     # (6, 30)
    c_j = np.median(R, axis=0)
    s_j = np.median(np.abs(R - c_j), axis=0)
    s_j = np.maximum(s_j, 1e-9)

    def T_of(mg):
        return float(np.max(np.maximum(mg - c_j, 0.0) / s_j))

    cal_T = [T_of(mg) for mg in mgs(cal_paths)]
    test_T = [T_of(mg) for mg in mgs(test_paths)]
    cal_s = [sc(e) for e in cal]
    qT = conformal_q(cal_T, 0.20)
    qC = conformal_q([max(a, b) for a, b in zip(cal_s, cal_T)], 0.20)
    t_pass = float(np.mean([t <= qT for t in test_T]))
    c_pass = float(np.mean([max(sc(e), t) <= qC
                            for e, t in zip(test, test_T)]))
    joint_pass = float(np.mean(
        [t <= qT and sc(e) <= q20 for t, e in zip(test_T, test)]))
    print('qT %.3f | qC %.3f | T-pass %.3f | combined-pass %.3f | '
          'naive-joint %.3f' % (qT, qC, t_pass, c_pass, joint_pass),
          flush=True)

    sd0 = torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                     map_location=device)
    th_m = ctx['margins_of'](sd0)
    bot, top = record_subsets(th_m, k=10)
    th_objs = {o: obj_val(th_m, o, bot, top) for o in ('O1', 'O2', 'O3')}
    start_mg = mgs(starts)

    def t_from_sd(sd):
        return T_of(ctx['margins_of'](sd))

    def sc_T(sd):
        return t_from_sd(sd)

    def sc_joint(sd):
        mm = ctx['eval_full'](sd)
        return (0.0 if (sc(mm) <= q20 and t_from_sd(sd) <= qT)
                else 1.0)

    ck = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)
    out = {'qT': float(qT), 'qC': float(qC),
           'honest_T_pass': t_pass,
           'honest_combined_pass': c_pass,
           'honest_naive_joint_pass': joint_pass, 'arms': {}}
    out['arms']['T_only'] = run_attack(
        model, xF, yF, bot, top, ctx['margins_of'], sc_T, qT,
        cli.outdir, ck, 'tdef', starts, start_mg, th_objs,
        objs=('O2',), n_starts=n_starts, iters=iters)
    out['arms']['naive_joint'] = run_attack(
        model, xF, yF, bot, top, ctx['margins_of'], sc_joint, 0.5,
        cli.outdir, ck, 'tjoint', starts, start_mg, th_objs,
        objs=('O2',), n_starts=n_starts, iters=iters)

    def sc_C(sd):
        mm = ctx['eval_full'](sd)
        return max(sc(mm), t_from_sd(sd))

    out['arms']['combined'] = run_attack(
        model, xF, yF, bot, top, ctx['margins_of'], sc_C, qC,
        cli.outdir, ck, 'tcomb', starts, start_mg, th_objs,
        objs=('O2',), n_starts=n_starts, iters=iters)

    t_best = out['arms']['T_only']['O2']['best_gain_pass']
    c_best = out['arms']['combined']['O2']['best_gain_pass']
    n_best = out['arms']['naive_joint']['O2']['best_gain_pass']
    if c_best < GATE:
        v = 'COMBINED_CLOSES'
        why = ('single calibrated combined score max(s,T) closes O2 '
               '(best gp %.3f; T-alone %.3f)' % (c_best, t_best))
    elif t_best < GATE:
        v = 'PARTIAL'
        why = 'T closes but combined leaks'
    else:
        v = 'T_LEAKS'
        why = 'direct per-record score admits passing gain'
    out.update({'verdict': v, 'verdict_why': why})
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', v, '|', why, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True,
                    choices=('reanalysis', 'bias', 'tdefense'))
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'r1-' + cli.arm + '-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = cli.outdir or os.path.join('research_outputs', 'uw',
                                            tag)
    os.makedirs(cli.outdir, exist_ok=cli.outdir is not None)
    print('R1 arm', cli.arm, '->', cli.outdir, flush=True)
    if cli.arm == 'reanalysis':
        run_reanalysis(cli)
    elif cli.arm == 'bias':
        run_bias(cli)
    else:
        run_tdefense(cli)


if __name__ == '__main__':
    main()

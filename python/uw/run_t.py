"""UW E-T: new-capability search arms (frozen prereg
docs/plans/2026-10-03-uw-t-capability-search-preregistration.md).

  t1 : ensemble-intersection ascent -- constraint
       max_k s_k(theta)/q_k <= 1 over K bootstrap estimators of the
       attacker's own pooled audit; judged by the true audit; commit
       = best-O2 final; T3 selection rules evaluated on the saved
       finals (quantile-anchored).
  t2 : multi-class deletion cell -- F = 15 cat + 15 truck, same
       theta0, 30 honest ft models, same audit structure; O2 over
       the global bottom-10; 2-parameter class-bias baseline.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np

from uw.run_ey import (GATE, METRICS, K, conformal_q, gain_of,
                       per_record_margin, record_subsets)
from uw.run_ez import (apply_delta, make_context, obj_val, run_attack,
                       setup_c10)
from uw.run_r1 import CACHE, RUN3, load_audit
from uw.run_r2 import theta_ctx
from uw.run_s import pooled_est, ascend_under
from uw.run_ew3b import LR0, lr_update

M = ('m1_acc', 'm2_catacc', 'm3_Floss', 'm4_KS', 'm5_adv', 'm6_fm')


def boot_estimators(entries, K, rng, alpha=0.20):
    ests = []
    for _ in range(K):
        idx = rng.choice(len(entries), len(entries), replace=True)
        ests.append(pooled_est([entries[i] for i in idx]))
    return ests


def run_t1(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 2 if cli.smoke else 60
    ctx = setup_c10(device)
    fit, cal, test, c, sig, sc_true, q = load_audit()
    q20 = q[0.20]
    _, th_m, bot, top = theta_ctx(ctx)
    th_o2 = obj_val(th_m, 'O2', bot, top)
    ens = json.load(open(CACHE))['ens_mm']
    own = ens['e3'] + ens['e4']
    starts = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    mgs = [ctx['margins_of'](torch.load(p, map_location=device))
           for p in starts]
    o2s = [obj_val(m, 'O2', bot, top) for m in mgs]
    med, sel = float(np.median(o2s)), max(o2s)
    ck = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)
    out = {}

    def run_intersection(ests, tag):
        def sfun(sd):
            mm = ctx['eval_full'](sd)
            return max(sc(mm) / qe for sc, qe in ests)
        rows = []
        for j, p in enumerate(starts[:(2 if cli.smoke else 6)]):
            sd = torch.load(p, map_location=device)
            fin, best = ascend_under(ctx, sd, sfun, 1.0, bot, top,
                                     iters)
            ts = sc_true(ctx['eval_full'](fin))
            mgf = ctx['margins_of'](fin)
            torch.save(fin, os.path.join(ck, '{}_s{}.pt'.format(tag, j)))
            rows.append({'j': j, 'best': best,
                         'true_pass': bool(ts <= q20),
                         'gain': float(gain_of(best, sel, med, th_o2)),
                         'w2c': int(((mgf > 0) & ~(mgs[j] > 0)).sum())})
            print(' t1', tag, j, 'gain %.3f pass %s w2c %d'
                  % (rows[-1]['gain'], rows[-1]['true_pass'],
                     rows[-1]['w2c']), flush=True)
        bestrow = max(rows, key=lambda r: r['best'])
        return {'per_start': rows, 'commit': bestrow}

    rng = np.random.RandomState(48200)
    main_ests = [(lambda e: pooled_est(own)[0](e)),
                 pooled_est(own)[1]]  # placeholder, replaced below
    sc_p, q_p = pooled_est(own)
    K_main = 5
    main_ests = boot_estimators(own, K_main, rng)
    out['main'] = run_intersection(main_ests, 'ix')
    frac_pass = np.mean([r['true_pass'] for r in out['main'][
        'per_start']])
    out['all_pass_frac'] = float(frac_pass)

    boot = []
    for b in range(1 if cli.smoke else 3):
        rng2 = np.random.RandomState(48300 + b)
        ests = boot_estimators(own, K_main, rng2)
        r = run_intersection(ests, 'ixb{}'.format(b))
        r['commit_ok'] = bool(r['commit']['true_pass']
                              and r['commit']['gain'] >= GATE)
        boot.append(r)
    out['bootstrap'] = [{'commit_ok': r['commit_ok'],
                         'commit_gain': r['commit']['gain']}
                        for r in boot]
    frac = (sum(1 for r in boot if r['commit_ok']) / len(boot))
    commit_ok = bool(out['main']['commit']['true_pass']
                     and out['main']['commit']['gain'] >= GATE)
    if commit_ok and frac >= 2 / 3:
        v = 'SUBMISSION_ESTABLISHED'
    elif commit_ok:
        v = 'T1_PARTIAL_BOOT_FRAGILE'
    elif out['main']['commit']['gain'] < GATE:
        v = 'T1_TIGHT_FAIL'
    else:
        v = 'T1_LEAK_FAIL'
    # T3: quantile-anchored selection on saved intersection finals
    sc_est, q_est = pooled_est(own)
    est_scores = [sc_est(e) for e in own]
    thr = float(np.quantile(est_scores, 0.9))
    sel_rows = []
    for j in range(len(out['main']['per_start'])):
        fp = os.path.join(ck, 'ix_s{}.pt'.format(j))
        fin = torch.load(fp, map_location=device)
        sel_rows.append({'j': j,
                         'est_s': float(sc_est(ctx['eval_full'](fin))),
                         'pass': out['main']['per_start'][j][
                             'true_pass'],
                         'gain': out['main']['per_start'][j]['gain']})
    elig = [r for r in sel_rows if r['est_s'] <= thr]
    if elig:
        pick = max(elig, key=lambda r: r['gain'])
        out['t3_quantile_rule'] = pick
    else:
        out['t3_quantile_rule'] = None
    out['verdict'] = v
    out['boot_frac'] = frac
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', v, '| boot frac %.2f | t3' % frac,
          out['t3_quantile_rule'], flush=True)


def run_t2(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 2 if cli.smoke else 60
    n_ft = 2 if cli.smoke else 6
    spec = dict(dataset='CIFAR10', arch='ResNet18', nc=10, cls=3,
                pool_scheme='multi', t0=0, ens0=48111,
                t0_src=os.path.join(RUN3, 'ckpt', 'theta0.pt'))
    ctx = make_context(spec, device)
    keep = ctx['keep']
    ck = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)
    sd0 = torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                     map_location=device)
    th_m = ctx['margins_of'](sd0)
    bot, top = record_subsets(th_m, k=10)
    th_objs = {o: obj_val(th_m, o, bot, top)
               for o in ('O1', 'O2', 'O3')}
    n_ens = 5
    n_per = n_ft
    ens_mm, ens_paths, ens_sds = {}, {}, {}
    from uw.run_ez import cached_train
    for e in range(n_ens):
        tag = 'z{}'.format(e + 1)
        ens_mm[tag], ens_paths[tag], ens_sds[tag] = [], [], []
        for j in range(n_per):
            seed = 48111 + e * 10 + j
            name = '{}_ft_s{}'.format(tag, seed)
            sd, mm = cached_train(
                ctx, ck, name, keep, seed, 1, init=sd0, cosine=False,
                eval_fn=ctx['eval_full'], outdir=cli.outdir)
            ens_sds[tag].append(sd)
            ens_mm[tag].append(mm)
            ens_paths[tag].append(os.path.join(ck, name + '.pt'))
        print(' t2 ensemble', tag, 'fm med %.2f'
              % float(np.median([m['m6_fm'] for m in ens_mm[tag]])),
              flush=True)
    c = {m: float(np.median([e[m] for e in ens_mm['z1']]))
         for m in METRICS}
    sig = {m: float(np.median(np.abs(
        np.array([e[m] for e in ens_mm['z1']]) - c[m])))
        for m in METRICS}

    def sc(mm):
        return max(abs(mm[m] - c[m]) / max(sig[m], 1e-12)
                   for m in METRICS)
    cal = [m for t in ('z2', 'z3') for m in ens_mm[t]]
    q20 = conformal_q([sc(e) for e in cal], 0.20)
    honest = [m for t in ('z2', 'z3', 'z4', 'z5')
              for m in ens_mm.get(t, [])]
    out = {'q20': float(q20),
           'honest_pass': float(np.mean([sc(e) <= q20
                                         for e in honest]))}
    print(' t2 q20 %.3f honest %.3f'
          % (q20, out['honest_pass']), flush=True)
    t0, t1 = 'z4', 'z5'
    mgs = {t: [ctx['margins_of'](sd) for sd in ens_sds[t]]
           for t in (t0, t1)}

    def score_sd(sd):
        return sc(ctx['eval_full'](sd))

    out['oracle_primary'] = run_attack(
        ctx['model'], ctx['xF'], ctx['yF'], bot, top, ctx['margins_of'],
        score_sd, q20, cli.outdir, ck, 'mc', ens_paths[t0], mgs[t0],
        th_objs, objs=('O1', 'O2') if not cli.smoke else ('O2',),
        n_starts=1 if cli.smoke else 3, iters=iters)
    out['oracle_repl_O2'] = run_attack(
        ctx['model'], ctx['xF'], ctx['yF'], bot, top, ctx['margins_of'],
        score_sd, q20, cli.outdir, ck, 'mcR', ens_paths[t1], mgs[t1],
        th_objs, objs=('O2',), n_starts=1 if cli.smoke else 3,
        iters=iters)
    # 2-parameter class bias baseline (cat=3, truck=9)
    o2s = [obj_val(m, 'O2', bot, top) for m in mgs[t0]]
    med, sel = float(np.median(o2s)), max(o2s)
    passing = [p for p, mm in zip(ens_paths[t0], ens_mm[t0])
               if sc(mm) <= q20][:3 if not cli.smoke else 1]
    bias = []
    for p in passing:
        bb, bv = None, None
        for bc in (0.5, 1, 2):
            for bt in (0.5, 1, 2):
                sd2 = {k: v.clone() for k, v in torch.load(
                    p, map_location=device).items()}
                sd2['fc.bias'][3] += bc
                sd2['fc.bias'][9] += bt
                mm2 = ctx['eval_full'](sd2)
                if sc(mm2) <= q20:
                    v = obj_val(ctx['margins_of'](sd2), 'O2', bot, top)
                    if bv is None or v > bv:
                        bb, bv = (bc, bt), v
        bias.append({'start': os.path.basename(p), 'b_max': bb,
                     'gain': (float(gain_of(bv, sel, med,
                                            th_objs['O2']))
                              if bv is not None else None)})
        print(' t2 bias', os.path.basename(p), bb, bias[-1]['gain'],
              flush=True)
    out['bias'] = bias
    o2_gp = out['oracle_primary']['O2']['best_gain_pass']
    o2_repl = out['oracle_repl_O2']['O2']['best_gain_pass']
    bmax = max([b['gain'] for b in bias
                if b['gain'] is not None] or [-1e9])
    if o2_gp >= GATE and o2_repl >= GATE and bmax < GATE:
        v = 'MATERIAL_DIFFERENTIAL'
    elif o2_gp >= GATE and o2_repl >= GATE:
        v = 'T2_NO_DIFFERENTIAL'
    elif max(out['oracle_primary'].get(
            'O1', {'best_gain_pass': -1e9})['best_gain_pass'],
            o2_gp) >= GATE:
        v = 'T2_PARTIAL'
    else:
        v = 'T2_NULL'
    out['verdict'] = v
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', v, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=('t1', 't2'))
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 't-' + cli.arm + '-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = cli.outdir or os.path.join('research_outputs', 'uw',
                                            tag)
    os.makedirs(cli.outdir, exist_ok=cli.outdir is not None)
    print('E-T arm', cli.arm, '->', cli.outdir, flush=True)
    dict(t1=run_t1, t2=run_t2)[cli.arm](cli)


if __name__ == '__main__':
    main()

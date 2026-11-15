"""UW R4: second-calibration-draw replication + T at tight alpha +
conservative selection rules (frozen in
docs/plans/2026-10-03-uw-r3-solidify-preregistration.md, R4 block).

Arms:
  calib2extra : class-bias baseline + combined T score on the
                calib2 cell (run_ez --arm calib2 first).
  talpha      : combined max(s,T) re-calibrated at alpha in
                {0.05, 0.10} on the primary calibration; honest
                pass + O2 attack.
  conselsel   : conservative SELECTION rules on the R2-nocon
                archived finals (lowest est-score final; median-O2
                final), judged by the true audit.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np

from uw.run_ey import GATE, conformal_q, gain_of, record_subsets
from uw.run_ez import obj_val, run_attack, setup_c10
from uw.run_ey2 import mk_audit
from uw.run_r1 import CACHE, RUN3, load_audit
from uw.run_r2 import ey2_dir, theta_ctx
from uw.run_r3 import tfit


def calib2_dir():
    return sorted(glob.glob('research_outputs/uw/r4a-calib2-2*'))[-1]


def run_calib2extra(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    d = calib2_dir()
    v = json.load(open(os.path.join(d, 'verdict.json')))

    def mm_of(tag):
        return [json.load(open(p)) for p in sorted(glob.glob(
            os.path.join(d, '{}_ft_s*_mm.json'.format(tag))))]

    fit_mm = mm_of('z1')
    cal_mm = mm_of('z2') + mm_of('z3')
    M = ('m1_acc', 'm2_catacc', 'm3_Floss', 'm4_KS', 'm5_adv', 'm6_fm')
    c = {m: float(np.median([e[m] for e in fit_mm])) for m in M}
    sig = {m: float(np.median(np.abs(
        np.array([e[m] for e in fit_mm]) - c[m]))) for m in M}

    def sc(e):
        return max(abs(e[m] - c[m]) / max(sig[m], 1e-12) for m in M)
    q20 = conformal_q([sc(e) for e in cal_mm], 0.20)
    _, th_m, bot, top = theta_ctx(ctx)
    th_o2 = obj_val(th_m, 'O2', bot, top)
    z4p = sorted(glob.glob(os.path.join(d, 'ckpt', 'z4_ft_s*.pt')))
    z4mg = [ctx['margins_of'](torch.load(p, map_location=device))
            for p in z4p]
    o2s = [obj_val(m, 'O2', bot, top) for m in z4mg]
    med, sel = float(np.median(o2s)), max(o2s)
    passing = [p for p in z4p
               if sc(ctx['eval_full'](torch.load(p,
                                                 map_location=device)))
               <= q20][:3]
    out = {'q20': float(q20), 'bias': [], 'T': {}}
    for p in passing:
        bb, bv = None, None
        for b in ((1.0, 4.0) if cli.smoke else (0.5, 1, 2, 4, 8)):
            sd2 = {k: v2.clone() for k, v2 in torch.load(
                p, map_location=device).items()}
            sd2['fc.bias'][3] += b
            if sc(ctx['eval_full'](sd2)) <= q20:
                bb, bv = b, obj_val(ctx['margins_of'](sd2), 'O2',
                                    bot, top)
        out['bias'].append({'start': os.path.basename(p), 'b_max': bb,
                            'gain': (float(gain_of(bv, sel, med, th_o2))
                                     if bv is not None else None)})
        print(' calib2 bias', os.path.basename(p), bb,
              out['bias'][-1]['gain'], flush=True)
    fit_paths = sorted(glob.glob(os.path.join(d, 'ckpt',
                                              'z1_ft_s*.pt')))
    c_j, s_j = tfit(ctx, fit_paths)

    def T_sd(sd):
        mg = ctx['margins_of'](sd)
        return float(np.max(np.maximum(mg - c_j, 0.0) / s_j))

    cal_s = [sc(e) for e in cal_mm]
    qC = conformal_q([max(a, T_sd(torch.load(p, map_location=device)))
                      for a, p in zip(cal_s, sorted(glob.glob(
                          os.path.join(d, 'ckpt', 'z2_ft_s*.pt')))
                          + sorted(glob.glob(os.path.join(
                              d, 'ckpt', 'z3_ft_s*.pt'))))], 0.20)
    test_p = z4p + sorted(glob.glob(os.path.join(d, 'ckpt',
                                                 'z5_ft_s*.pt')))
    ho = float(np.mean([
        max(sc(ctx['eval_full'](torch.load(p, map_location=device))),
            T_sd(torch.load(p, map_location=device))) <= qC
        for p in test_p]))
    out['T'] = {'qC': float(qC), 'honest': ho}

    def sc_C(sd):
        return max(sc(ctx['eval_full'](sd)), T_sd(sd))

    ck = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)
    out['T']['attack'] = run_attack(
        ctx['model'], ctx['xF'], ctx['yF'], bot, top,
        ctx['margins_of'], sc_C, qC, cli.outdir, ck, 'c2comb',
        passing, z4mg[:len(passing)], {'O2': th_o2}, objs=('O2',),
        n_starts=(1 if cli.smoke else 3),
        iters=(2 if cli.smoke else 60))
    out['verdict'] = ('CALIB2_T_LEAKS' if out['T']['attack']['O2'][
        'best_gain_pass'] >= GATE else 'CALIB2_T_CLOSES')
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('qC %.3f honest %.3f | VERDICT %s'
          % (qC, ho, out['verdict']), flush=True)


def run_talpha(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    fit, cal, test, c, sig, sc, q = load_audit()
    _, th_m, bot, top = theta_ctx(ctx)
    th_objs = {'O2': obj_val(th_m, 'O2', bot, top)}
    e3 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    e3mg = [ctx['margins_of'](torch.load(p, map_location=device))
            for p in e3]
    passing = [p for p in e3
               if sc(ctx['eval_full'](torch.load(p,
                                                 map_location=device)))
               <= q[0.20]][:3]
    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-2*/ckpt/ft_k30_s*.pt'))
    c_j, s_j = tfit(ctx, fit_paths)

    def T_sd(sd):
        mg = ctx['margins_of'](sd)
        return float(np.max(np.maximum(mg - c_j, 0.0) / s_j))

    cal_paths = (sorted(glob.glob(
        'research_outputs/uw/r3p1-2*/ckpt/ft_aud_s*.pt'))
        + sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/e2_ft_s*.pt')))
    cal_pair = [(sc(e), T_sd(torch.load(p, map_location=device)))
                for e, p in zip(cal, cal_paths)]
    out = {}
    ck = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)
    for a in ((0.10,) if cli.smoke else (0.05, 0.10)):
        qC = conformal_q([max(x, y) for x, y in cal_pair], a)
        ho = float(np.mean([
            max(sc(e), T_sd(torch.load(p, map_location=device)))
            <= qC for e, p in zip(
                test, e3 + sorted(glob.glob(
                    'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt')))]))

        def sc_C(sd):
            return max(sc(ctx['eval_full'](sd)), T_sd(sd))

        res = run_attack(
            ctx['model'], ctx['xF'], ctx['yF'], bot, top,
            ctx['margins_of'], sc_C, qC, cli.outdir, ck,
            'ta' + str(a).replace('.', ''), passing,
            e3mg[:len(passing)], th_objs, objs=('O2',),
            n_starts=(1 if cli.smoke else 3),
            iters=(2 if cli.smoke else 60))
        out[str(a)] = {'qC': float(qC), 'honest': ho, 'attack': res}
        print(' alpha', a, 'qC %.3f honest %.3f best_gp %.3f'
              % (qC, ho, res['O2']['best_gain_pass']), flush=True)
    a5, a1 = out.get('0.05'), out.get('0.1')
    closed = [d for d in (a5, a1) if d and d['attack']['O2'][
        'best_gain_pass'] < GATE and d['honest'] >= 10 / 12]
    out['verdict'] = ('T_ALPHA_CLOSES' if closed else 'T_ALPHA_LEAKS')
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', out['verdict'], flush=True)


def run_conselsel(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]
    ens = json.load(open(CACHE))['ens_mm']
    est = mk_audit(ens['e3'], ens['e4'])
    _, th_m, bot, top = theta_ctx(ctx)
    nocon = sorted(glob.glob('research_outputs/uw/r2-nocon-2*'))[-1]
    out = {}
    for lam in ('0.4', '0.6', '0.8'):
        pref = 'lam' + str(float(lam)).replace('.', '')
        rows = []
        for j in range(3):
            fp = os.path.join(nocon, '{}_O2_s{}.json'.format(pref, j))
            cp = os.path.join(nocon, 'ckpt',
                              '{}_O2_s{}.pt'.format(pref, j))
            if not (os.path.exists(fp) and os.path.exists(cp)):
                continue
            rec = json.load(open(fp))
            fin = torch.load(cp, map_location=device)
            rows.append({'j': j, 'best': rec['best'],
                         'est_s': float(est[3](ctx['eval_full'](fin))),
                         'true_pass': rec.get('true_pass')})
        if not rows:
            continue
        low = min(rows, key=lambda r: r['est_s'])
        medr = sorted(rows, key=lambda r: r['best'])[len(rows) // 2]
        out[lam] = {'lowest_est': low, 'median_o2': medr}
        print(' lam', lam, 'lowest-est pass', low['true_pass'],
              'best', round(low['best'], 2), '| median pass',
              medr['true_pass'], flush=True)
    rescued = any(v['lowest_est']['true_pass'] or
                  v['median_o2']['true_pass'] for v in out.values())
    out['verdict'] = ('SELECTION_RESCUED' if rescued
                      else 'SELECTION_NOT_RESCUED')
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', out['verdict'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True,
                    choices=('calib2extra', 'talpha', 'conselsel'))
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'r4-' + cli.arm + '-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = cli.outdir or os.path.join('research_outputs', 'uw',
                                            tag)
    os.makedirs(cli.outdir, exist_ok=cli.outdir is not None)
    print('R4 arm', cli.arm, '->', cli.outdir, flush=True)
    dict(calib2extra=run_calib2extra, talpha=run_talpha,
         conselsel=run_conselsel)[cli.arm](cli)


if __name__ == '__main__':
    main()

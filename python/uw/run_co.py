"""UW E-Co: audit-coalition attack (frozen prereg
docs/plans/2026-10-04-uw-co-audit-coalition-*.md).

Joint score J(theta) = max(s_ft/5.245, s_rt/4.647, C/10.465,
cap_pen) <= 1 over the four calibrated audit families; full-model
ascent, last-layer ascent, and class bias under the SAME joint
constraint.
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from uw.run_ey import GATE, conformal_q, gain_of, record_subsets
from uw.run_ez import obj_val, setup_c10
from uw.run_ey2 import mk_audit
from uw.run_r1 import RUN3, load_audit
from uw.run_r2 import theta_ctx
from uw.run_r3 import tfit
from uw.run_s import ascend_under, ascend_under_ll
from uw.run_u import bias_bisect

RUN6 = 'research_outputs/uw/ew6-20260925T213308'


def run_co(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 4 if cli.smoke else 60
    ctx = setup_c10(device)
    # family 1: ft conformal
    fit, cal, test, c, sig, sc_ft, q = load_audit()
    q_ft = q[0.20]
    # family 2: symmetric band on raw fm (same fit family)
    fm_med = float(np.median([e['m6_fm'] for e in fit]))
    fm_b = float(max(abs(e['m6_fm'] - fm_med) for e in fit))
    sc_sb = q_sb = None  # predicate below
    # family 3: combined T score
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
    cal_pair = [(sc_ft(e), T_sd(torch.load(p, map_location=device)))
                for e, p in zip(cal, cal_paths)]
    q_C = conformal_q([max(a, b) for a, b in cal_pair], 0.20)
    # family 4: one-sided cap
    fm_max = float(max(e['m6_fm'] for e in fit))
    print('coalition: q_ft %.3f q_C %.3f fm_max %.3f fm_b %.3f'
          % (q_ft, q_C, fm_max, fm_b), flush=True)

    def J(sd):
        mm = ctx['eval_full'](sd)
        cap_pen = 0.0 if mm['m6_fm'] <= fm_max else 1.0
        sb_pen = 0.0 if abs(mm['m6_fm'] - fm_med) <= fm_b else 1.0
        return max(sc_ft(mm) / q_ft,
                   max(sc_ft(mm), T_sd(sd)) / q_C, cap_pen, sb_pen)

    _, th_m, bot, top = theta_ctx(ctx)
    th_o2 = obj_val(th_m, 'O2', bot, top)
    starts = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    mgs = [ctx['margins_of'](torch.load(p, map_location=device))
           for p in starts]
    o2s = [obj_val(m, 'O2', bot, top) for m in mgs]
    med, sel = float(np.median(o2s)), max(o2s)
    n_st = 2 if cli.smoke else 6
    out = {'full': [], 'll': [], 'bias': []}
    for j, p in enumerate(starts[:n_st]):
        sd = torch.load(p, map_location=device)
        fin, best = ascend_under(ctx, sd, J, 1.0, bot, top, iters)
        o2v = obj_val(ctx['margins_of'](fin), 'O2', bot, top)
        out['full'].append({'j': j, 'J': float(J(fin)),
                            'pass': bool(J(fin) <= 1.0),
                            'gain': float(gain_of(o2v, sel, med, th_o2))})
        print(' co full', j, 'J %.3f pass %s gain %.3f'
              % (out['full'][-1]['J'], out['full'][-1]['pass'],
                 out['full'][-1]['gain']), flush=True)
    for j, p in enumerate(starts[:n_st]):
        sd = torch.load(p, map_location=device)
        fin, best = ascend_under_ll(ctx, sd, J, 1.0, bot, top, iters)
        o2v = obj_val(ctx['margins_of'](fin), 'O2', bot, top)
        out['ll'].append({'j': j, 'pass': bool(J(fin) <= 1.0),
                          'gain': float(gain_of(o2v, sel, med, th_o2))})
        print(' co ll', j, 'pass %s gain %.3f'
              % (out['ll'][-1]['pass'], out['ll'][-1]['gain']),
              flush=True)
    passing0 = [j for j, p in enumerate(starts) if J(torch.load(
        p, map_location=device)) <= 1.0][:n_st]
    for j in passing0:
        sd = torch.load(starts[j], map_location=device)
        # grid bias under J
        bb, bv = None, None
        for b in (0.5, 1, 2, 4, 8):
            sd2 = {k: v.clone() for k, v in sd.items()}
            sd2['fc.bias'][3] += b
            if J(sd2) <= 1.0:
                bb, bv = b, obj_val(ctx['margins_of'](sd2), 'O2',
                                    bot, top)
        out['bias'].append({
            'j': j, 'b_max': bb,
            'gain': (float(gain_of(bv, sel, med, th_o2))
                     if bv is not None else None),
            'pass': bb is not None})
        print(' co bias', j, bb, out['bias'][-1]['gain'], flush=True)
    f_gp = max((r['gain'] if r['pass'] else -1e9)
               for r in out['full']) if out['full'] else -1e9
    ll_bp = max([r['gain'] for r in out['ll'] if r['pass']] or [-1e9])
    b_bp = max([r['gain'] for r in out['bias']
                if r['pass'] and r['gain'] is not None] or [-1e9])
    out['summary'] = {'full_gp': f_gp, 'll_bp': ll_bp, 'bias_bp': b_bp}
    if f_gp >= GATE and ll_bp < GATE and b_bp < GATE:
        v = 'DIFFERENTIAL'
    elif f_gp < GATE and ll_bp < GATE and b_bp < GATE:
        v = 'JOINT_NULL'
    else:
        v = 'NO_DIFFERENTIAL'
    out['verdict'] = v
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', v, '| full %.3f ll %.3f bias %.3f'
          % (f_gp, ll_bp, b_bp), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'co-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(cli.outdir, exist_ok=False)
    print('E-Co ->', cli.outdir, flush=True)
    run_co(cli)


if __name__ == '__main__':
    main()

"""UW E-Cr: calibration-resolution frontier (frozen prereg
docs/plans/2026-10-04-uw-cr-calibration-resolution-*.md).

Fix the combined score C = max(s, T) (fit = ew9f, unchanged); vary
ONLY the calibration-set size n_cal in {12, 30, 60, 120} from 120
freshly trained ft models.  At each size: threshold qC@0.20,
held-out honest pass (e3+e4), and three channels under C <= qC
(class bias grid / last-layer ascent / full ascent) from e3's
three audit-passing starts.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np

from uw.run_ey import GATE, conformal_q, gain_of, record_subsets
from uw.run_ez import cached_train, obj_val, setup_c10
from uw.run_r1 import load_audit
from uw.run_r2 import theta_ctx
from uw.run_r3 import tfit
from uw.run_s import ascend_under, ascend_under_ll

CAL_SEEDS = [49111 + i for i in range(120)]
SIZES = (12, 30, 60, 120)
START_IDX = (1, 3, 4)          # e3 starts passing the original audit


def run_cr(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 4 if cli.smoke else 60
    ctx = setup_c10(device)
    fit, cal, test, c, sig, sc_ft, q = load_audit()
    _, th_m, bot, top = theta_ctx(ctx)
    th_o2 = obj_val(th_m, 'O2', bot, top)
    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-2*/ckpt/ft_k30_s*.pt'))
    c_j, s_j = tfit(ctx, fit_paths)

    def T_sd(sd):
        mg = ctx['margins_of'](sd)
        return float(np.max(np.maximum(mg - c_j, 0.0) / s_j))

    def C_sd(sd):
        return max(sc_ft(ctx['eval_full'](sd)), T_sd(sd))

    ck = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)
    keep = ctx['keep']
    cal_ck = []
    n_train = (4 if cli.smoke else 120)
    for i, seed in enumerate(CAL_SEEDS[:n_train]):
        sd, mm = cached_train(
            ctx, ck, 'cr_s{}'.format(seed), keep, seed, 1,
            init=torch.load(os.path.join(
                'research_outputs/uw/ew3-20260925T113900', 'ckpt',
                'theta0.pt'), map_location=device),
            cosine=False, eval_fn=ctx['eval_full'],
            outdir=cli.outdir)
        cal_ck.append(sd)
        if (i + 1) % 20 == 0:
            print(' cal models', i + 1, flush=True)
    cal_C = [C_sd(sd) for sd in cal_ck]
    starts = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    e34 = starts + sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))
    honest_C = [C_sd(torch.load(p, map_location=device)) for p in e34]
    sds = [torch.load(starts[j], map_location=device)
           for j in START_IDX]
    mgs = [ctx['margins_of'](torch.load(p, map_location=device))
           for p in starts]
    o2s = [obj_val(m, 'O2', bot, top) for m in mgs]
    med, sel = float(np.median(o2s)), max(o2s)
    out = {'sizes': {}}
    for n in ((2,) if cli.smoke else SIZES):
        qC = conformal_q(cal_C[:n], 0.20)
        ho = float(np.mean([h <= qC for h in honest_C]))
        rec = {'qC': float(qC), 'honest12': ho}
        print(' size', n, 'qC %.3f honest %.3f' % (qC, ho), flush=True)

        def J(sd):
            return C_sd(sd)
        for chan in ('bias', 'll', 'full'):
            rows = []
            for sd in sds:
                if chan == 'bias':
                    bb, bv = None, None
                    for b in (0.25, 0.5, 1, 2, 4):
                        sd2 = {k: v.clone() for k, v in sd.items()}
                        sd2['fc.bias'][3] += b
                        if C_sd(sd2) <= qC:
                            bb, bv = b, obj_val(
                                ctx['margins_of'](sd2), 'O2', bot, top)
                    rows.append({'pass': bb is not None,
                                 'gain': (float(gain_of(
                                     bv, sel, med, th_o2))
                                     if bv is not None else None)})
                else:
                    fn = (ascend_under if chan == 'full'
                          else ascend_under_ll)
                    fin, best = fn(ctx, sd, J, qC, bot, top, iters)
                    rows.append({'pass': bool(C_sd(fin) <= qC),
                                 'gain': float(gain_of(
                                     best, sel, med, th_o2))})
            bp = max([r['gain'] for r in rows
                      if r['pass'] and r['gain'] is not None] or [-1e9])
            rec[chan] = {'per_start': rows, 'best_passing': bp}
            print('  ', chan, 'bp %.3f' % bp, flush=True)
        out['sizes'][str(n)] = rec
    diff = [n for n, r in out['sizes'].items()
            if r['full']['best_passing'] >= GATE
            and r['bias']['best_passing'] < GATE
            and r['ll']['best_passing'] < GATE
            and r['honest12'] >= 8 / 12]
    closed = [n for n, r in out['sizes'].items()
              if r['full']['best_passing'] < GATE
              and r['bias']['best_passing'] < GATE
              and r['ll']['best_passing'] < GATE
              and r['honest12'] >= 8 / 12]
    if diff:
        v = 'DIFFERENTIAL_AT_' + diff[-1]
    elif closed:
        v = 'DEFENSE_CLOSED_AT_' + closed[0]
    else:
        v = 'MIXED_CURVE'
    out['verdict'] = v
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', v, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'cr-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = cli.outdir or os.path.join('research_outputs', 'uw',
                                            tag)
    os.makedirs(cli.outdir, exist_ok=cli.outdir is not None)
    print('E-Cr ->', cli.outdir, flush=True)
    run_cr(cli)


if __name__ == '__main__':
    main()

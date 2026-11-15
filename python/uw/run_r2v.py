"""UW R2v: zero-training closure computations for the R2 verdict.

  randcommit : uniform-random-commit joint success rate over the
               six own-information finals (ey2 orc_e3_O2) and the
               nocon lambda finals: P(pass AND flips>=1) etc.,
               computed only from pre-committed finals (no audit
               filtering).
  tbehavior  : forward-only behavioral table for all saved
               combined/T-defense finals (r1 tdefense, r2 c100loop,
               r3 trep, r4 calib2extra): O2 value, wrong-to-correct /
               correct-to-wrong flips vs own start, per-class max
               damage.
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from uw.run_ey import record_subsets
from uw.run_ez import obj_val, setup_c10
from uw.run_r2 import ey2_dir, theta_ctx
from uw.run_r3 import perclass_diff


def flips_vs_start(ctx, fin_sd, start_sd, bot):
    mg_f = ctx['margins_of'](fin_sd)
    mg_s = ctx['margins_of'](start_sd)
    return int(((mg_f > 0) & ~(mg_s > 0)).sum()), \
        int(((mg_s > 0) & ~(mg_f > 0)).sum()), mg_f, mg_s


def run_randcommit(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    _, th_m, bot, top = theta_ctx(ctx)
    starts = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    out = {}
    # arm 1: ey2 correctly-informed finals (own-info: fixed program
    # per start; uniform commit over 6)
    rows = []
    for j, p in enumerate(starts):
        fp = os.path.join(ey2_dir(), 'orc_e3_O2_s{}.json'.format(j))
        cp = os.path.join(ey2_dir(), 'ckpt',
                          'orc_e3_O2_s{}.pt'.format(j))
        if not os.path.exists(fp):
            continue
        rec = json.load(open(fp))
        w2c, c2w, _, _ = flips_vs_start(
            ctx, torch.load(cp, map_location=device),
            torch.load(p, map_location=device), bot)
        rows.append({'j': j, 'pass': rec.get('true_pass',
                                             rec.get('est_pass')),
                     'best': rec['best'], 'w2c': w2c, 'c2w': c2w})
    n = len(rows)
    out['reference_arm'] = {
        'per_final': rows,
        'uniform_commit_pass': sum(r['pass'] for r in rows) / n,
        'uniform_commit_joint(pass & w2c>=1)':
            sum(r['pass'] and r['w2c'] >= 1 for r in rows) / n,
        'construction_completed': n}
    print('ref finals:', [(r['pass'], r['w2c']) for r in rows],
          '| P(pass)=%.3f P(joint)=%.3f'
          % (out['reference_arm']['uniform_commit_pass'],
             out['reference_arm'][
                 'uniform_commit_joint(pass & w2c>=1)']), flush=True)
    # arm 2: nocon lambda finals (self-estimated constraint only)
    nocon = sorted(glob.glob('research_outputs/uw/r2-nocon-2*'))[-1]
    out['nocon'] = {}
    for lam in (0.4, 0.6, 0.8):
        pref = 'lam' + str(lam).replace('.', '')
        rows = []
        for j, p in enumerate(starts[:3]):
            fp = os.path.join(nocon, '{}_O2_s{}.json'.format(pref, j))
            cp = os.path.join(nocon, 'ckpt',
                              '{}_O2_s{}.pt'.format(pref, j))
            if not os.path.exists(fp):
                continue
            rec = json.load(open(fp))
            w2c, c2w, _, _ = flips_vs_start(
                ctx, torch.load(cp, map_location=device),
                torch.load(p, map_location=device), bot)
            rows.append({'j': j, 'pass': rec.get('true_pass'),
                         'w2c': w2c})
        if rows:
            out['nocon'][str(lam)] = {
                'per_final': rows,
                'uniform_commit_joint':
                    sum(r['pass'] and r['w2c'] >= 1
                        for r in rows) / len(rows)}
            print('nocon lam', lam, [(r['pass'], r['w2c'])
                                     for r in rows],
                  'joint=%.3f' % out['nocon'][str(lam)][
                      'uniform_commit_joint'], flush=True)
    with open(os.path.join(cli.outdir, 'randcommit.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)


T_SOURCES = [
    ('r1-tdefense-2*', ('tdef', 'tjoint', 'tcomb'), 'e3'),
    ('r2-c100loop-2*', ('c100comb',), 'c100z4'),
    ('r3-trep-2*', ('e4comb', 'rtcomb'), 'mixed'),
    ('r4-calib2extra-2*', ('c2comb',), 'calib2z4'),
]


def run_tbehavior(cli):
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    from uw.run_ez import CELLS, make_context
    ctx100 = make_context(CELLS['c100'], device)
    _, th_m, bot, top = theta_ctx(ctx)
    e3 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    e4 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))
    rtnew = [os.path.join(
        sorted(glob.glob('research_outputs/uw/ez5-retrainfix-2*'))[-1],
        'ckpt', 'rtnew_s{}.pt'.format(s))
        for s in (47301, 47302, 47303, 47304, 47305, 47306)]
    rows = []
    for pat, prefixes, pool in T_SOURCES:
        dirs = sorted(glob.glob('research_outputs/uw/' + pat))
        if not dirs:
            continue
        d = dirs[-1]
        for pref in prefixes:
            for cp in sorted(glob.glob(os.path.join(
                    d, 'ckpt', '{}_O2_s*.pt'.format(pref)))):
                j = int(os.path.basename(cp)[:-3].rsplit('s', 1)[1])
                if pool == 'e3':
                    stp, actx = e3[j], ctx
                elif pool == 'c100z4':
                    stp = sorted(glob.glob(os.path.join(
                        sorted(glob.glob(
                            'research_outputs/uw/ez7-c100b-2*'))[-1],
                        'ckpt', 'z4_ft_s*.pt')))[j]
                    actx = ctx100
                elif pool == 'calib2z4':
                    stp, actx = sorted(glob.glob(os.path.join(
                        sorted(glob.glob(
                            'research_outputs/uw/r4a-calib2-2*'))[-1],
                        'ckpt', 'z4_ft_s*.pt')))[j], ctx
                elif pref == 'e4comb':
                    stp, actx = e4[j], ctx
                else:
                    stp, actx = rtnew[j], ctx
                fin = torch.load(cp, map_location=device)
                st = torch.load(stp, map_location=device)
                w2c, c2w, mgf, mgs = flips_vs_start(actx, fin, st, bot)
                pc = perclass_diff(actx, fin, st)
                cls = 3 if actx is ctx else 0
                rows.append({'src': os.path.basename(d), 'pref': pref,
                             'j': j, 'w2c': w2c, 'c2w': c2w,
                             'abs_dO2': float(mgf[bot].mean()
                                              - mgs[bot].mean()),
                             'max_nonclass_dmg': float(min(
                                 pc[k] for k in range(len(pc))
                                 if k != cls))})
                print(' tb', pref, j, 'w2c', w2c, 'c2w', c2w,
                      'dO2 %.2f' % rows[-1]['abs_dO2'],
                      'maxdmg %.2f' % rows[-1]['max_nonclass_dmg'],
                      flush=True)
    with open(os.path.join(cli.outdir, 'tbehavior.json'), 'w') as fh:
        json.dump(rows, fh, indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True,
                    choices=('randcommit', 'tbehavior'))
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()
    tag = 'r2v-' + cli.arm + '-' + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = cli.outdir or os.path.join('research_outputs', 'uw',
                                            tag)
    os.makedirs(cli.outdir, exist_ok=False)
    dict(randcommit=run_randcommit, tbehavior=run_tbehavior)[cli.arm](
        cli)


if __name__ == '__main__':
    main()

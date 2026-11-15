"""Probe TIGHT (review ruling 15a, prereg
docs/plans/2026-11-03-review-probe-ab-prereg.md): can the encoder
re-optimize under tightened acceptance thresholds (0.99 q-hat,
0.95 q-hat)? ce_gen frozen protocol verbatim; only the acceptance
threshold changes. Starts: e3-s1 / e4-s0 / e4-s1 (strong CE starts).
Readout per arm/start: feasible steps, final s6 (true audit), AUC
gain CI (member-resampling, paired, Table VIII protocol).
Gates frozen in the prereg card: TIGHT-FOLLOWED / TIGHT-BLOCKS /
MIXED.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import setup_c10
from uw.run_ey import K
from uw.run_ew3 import f_deep
from uw.run_lfpr import ce_gen, channels
from uw.run_r1 import load_audit
from uw.run_gm import auc as rank_auc

ARMS = ((0.99, 't099'), (0.95, 't095'))
STARTS = (('e3-s1', 1), ('e4-s0', 0), ('e4-s1', 1))


def rank_ci(fm, bm, fe, be, n=2000, seed=53900):
    rs = np.random.RandomState(seed)
    base, fin = rank_auc(bm, be), rank_auc(fm, fe)
    diffs = [rank_auc(fm[rs.choice(len(fm), len(fm), replace=True)],
                      fe) - rank_auc(bm[rs.choice(len(fm), len(fm),
                                                  replace=True)], be)
            for _ in range(n)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return base, fin, float(lo), float(hi)


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    model = ctx['model']
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K)
    cons = np.sort(np.random.RandomState(52801).choice(cat_te, 300,
                                                       replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    mem = (ctx['xF'], ctx['yF'])
    from uw.run_mb import rec
    fx = rec(tset, fresh, device)
    cx = rec(tset, cons, device)

    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]

    def score_sd(sd):
        return sc(ctx['eval_full'](sd))

    e3 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    e4 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))
    start_paths = {'e3-s1': e3[1], 'e4-s0': e4[0], 'e4-s1': e4[1]}
    assert len(e3) == 6 and len(e4) == 6

    def neggrad_scores(sd):
        ch = channels(model, sd, *mem)
        return ch['neggrad']

    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'q20': q20, 'rows': []}
    for fac, aname in ARMS:
        thr = q20 * fac
        for tag, _ in STARTS:
            sp = start_paths[tag]
            sd0 = torch.load(sp, map_location=device)
            s6_0 = score_sd(sd0)
            bm = neggrad_scores(sd0)
            fe = channels(model, sd0, fx[0], fx[1])['neggrad']
            fin = ce_gen(model, mem, cx, score_sd, thr,
                         {'sd': sd0, 'n': 60})
            s6_f = score_sd(fin)
            fm = neggrad_scores(fin)
            ch_fin_f = channels(model, fin, fx[0], fx[1])
            base_auc, fin_auc, lo, hi = rank_ci(
                fm, bm, ch_fin_f['neggrad'], fe)
            row = {'arm': aname, 'factor': fac, 'start': tag,
                   'threshold': thr, 's6_base': s6_0, 's6_fin': s6_f,
                   'pass_tight': bool(s6_f <= thr),
                   'pass_true': bool(s6_f <= q20),
                   'auc_base': base_auc, 'auc_fin': fin_auc,
                   'ci': [lo, hi], 'ci_positive': bool(lo > 0)}
            out['rows'].append(row)
            print('%s %-6s s6 %.3f->%.3f (thr %.3f) pass_tight=%s '
                  'auc %.3f->%.3f CI[%.3f,%.3f]'
                  % (aname, tag, s6_0, s6_f, thr, row['pass_tight'],
                     base_auc, fin_auc, lo, hi), flush=True)

    followed = any(r['pass_tight'] and r['ci_positive']
                   for r in out['rows'])
    blocked = all((not r['pass_tight']) or (not r['ci_positive'])
                  for r in out['rows'])
    out['verdict'] = ('TIGHT-FOLLOWED' if followed
                      else 'TIGHT-BLOCKS' if blocked else 'MIXED')
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', out['verdict'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/probe-t-%s' % (
            time.strftime('%Y%m%dT%H%M%S'),)
    assert not os.path.exists(cli.outdir), cli.outdir
    os.makedirs(cli.outdir)
    run(cli)


if __name__ == '__main__':
    main()

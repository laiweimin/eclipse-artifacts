"""C-A8E P0: mechanism decomposition — why does the CE encoding
family pass the joint (covariance) rule while GR fails?
Zero training. Decomposes each final's six-metric deviation
vector d = m - c against the honest covariance eigensystem:
s_J, ||d||/scale, per-metric standardized components, and the
fraction of s_J carried by the top-k honest variance directions
(the directions the joint form DISCOUNTS) vs the tail.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import setup_c10
from uw.run_ey import METRICS
from uw.run_r1 import load_audit

CE_GLOB = ('research_outputs/uw/lfpr-20261009T152330/ckpt/ce_*.pt')
GR = ['research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s1.pt',
      'research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s3.pt',
      'research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s4.pt',
      'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s0.pt',
      'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s1.pt',
      'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s2.pt']
RHO = 0.5


def vec(e):
    return np.array([e[m] for m in METRICS], dtype=float)


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]

    X = np.stack([vec(e) for e in fit])
    c_j = X.mean(0)
    S = np.cov(X, rowvar=False)
    Sh = (1 - RHO) * S + RHO * np.diag(np.diag(S))
    evals, evecs = np.linalg.eigh(Sh)     # ascending
    Sinv = np.linalg.inv(Sh)
    # whitened eigenbasis: contribution of eigendir k to s_J
    # s_J = sum_k (v_k . d)^2 / lambda_k
    qJ = 16.4687  # frozen from run_a8joint (rho=0.5, same cal)

    ctx = setup_c10(device)

    def metrics_of(path):
        sd = torch.load(path, map_location=device)
        return ctx['eval_full'](sd)

    rows = []
    groups = ([('honest', e) for e in test]
              + [('ce', p) for p in sorted(glob.glob(CE_GLOB))]
              + [('gr', p) for p in GR])
    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'evals_ascending': [float(x) for x in evals],
           'qJ': qJ, 'rows': []}
    print('evals (asc): %s' % np.round(evals, 3), flush=True)
    print('%-8s %-6s %7s %7s %7s %7s %7s'
          % ('tag', 'grp', 's6', 'sJ', 'top1/2', 'top2/3', 'maxmetric'),
          flush=True)
    for i, (grp, obj) in enumerate(groups):
        e = obj if grp == 'honest' else metrics_of(obj)
        d = vec(e) - c_j
        sj = float(d @ Sinv @ d)
        comp = (evecs.T @ d) ** 2 / evals        # per-eigendir contrib
        frac12 = float(comp[np.argsort(evals)[::-1][:1]].sum()
                       / max(sj, 1e-12))
        frac23 = float(comp[np.argsort(evals)[::-1][:2]].sum()
                       / max(sj, 1e-12))
        std = np.abs(d) / np.array([sig[m] for m in METRICS])
        rows.append({'tag': '%s%d' % (grp, i), 'group': grp,
                     's6': float(sc(e)), 'sJ': sj,
                     'top1_frac': frac12, 'top2_frac': frac23,
                     'std_components': [float(x) for x in std]})
        print('%-8s %-6s %7.3f %7.3f %7.3f %7.3f %7.3f'
              % (rows[-1]['tag'], grp, rows[-1]['s6'], sj,
                 frac12, frac23, std.max()), flush=True)
        out['rows'] = rows
    ce = [r for r in rows if r['group'] == 'ce']
    gr = [r for r in rows if r['group'] == 'gr']
    ho = [r for r in rows if r['group'] == 'honest']
    summ = {'ce_sJ_mean': float(np.mean([r['sJ'] for r in ce])),
            'gr_sJ_mean': float(np.mean([r['sJ'] for r in gr])),
            'honest_sJ_mean': float(np.mean([r['sJ'] for r in ho])),
            'ce_top2_mean': float(np.mean([r['top2_frac']
                                           for r in ce])),
            'gr_top2_mean': float(np.mean([r['top2_frac']
                                           for r in gr])),
            'honest_top2_mean': float(np.mean([r['top2_frac']
                                               for r in ho])),
            'ce_headroom_vs_qJ': float(qJ - np.mean(
                [r['sJ'] for r in ce]))}
    out['summary'] = summ
    print('SUMMARY %s' % json.dumps(summ, indent=1), flush=True)
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/a8mech-%s' % (
            time.strftime('%Y%m%dT%H%M%S'),)
    os.makedirs(cli.outdir, exist_ok=True)
    run(cli)


if __name__ == '__main__':
    main()

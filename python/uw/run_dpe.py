"""C-DPE P0: dual headroom ratio decomposition (zero training).

For every honest start, original final, and adaptive final:
compute s6, s_J, and the dual headroom ratio
  rho_dual = min(q_hat - s6, q_J - s_J) / max(q_hat, q_J)
A high rho_dual means the model has room in BOTH constraint
directions; rho_dual ~ 0 means at least one constraint is binding.

Also: for each of the 6 lfpr honest starts, compute the dual
headroom after unconstrained CE encoding (from the persisted
marginal finals) to see where the encoder drives both scores.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import setup_c10
from uw.run_ey import METRICS, conformal_q
from uw.run_r1 import load_audit

RHO = 0.5
CE_ORIG = ('research_outputs/uw/lfpr-20261009T152330/ckpt/ce_*.pt')
GR_ORIG = ['research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s1.pt',
           'research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s3.pt',
           'research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s4.pt',
           'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s0.pt',
           'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s1.pt',
           'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s2.pt']
E3 = sorted(glob.glob('research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
E4 = sorted(glob.glob('research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))


def vec(e):
    return np.array([e[m] for m in METRICS], dtype=float)


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]
    X = np.stack([vec(e) for e in fit])
    c_j = X.mean(0)
    S = np.cov(X, rowvar=False)
    Sh = (1 - RHO) * S + RHO * np.diag(np.diag(S))
    Sinv = np.linalg.inv(Sh)
    qJ = conformal_q([float((vec(e) - c_j) @ Sinv @ (vec(e) - c_j))
                      for e in cal], 0.20)

    def scores(sd):
        e = ctx['eval_full'](sd)
        s6 = float(sc(e))
        d = vec(e) - c_j
        sj = float(d @ Sinv @ d)
        return s6, sj

    def rho_dual(s6, sj):
        return min(q20 - s6, qJ - sj) / max(q20, qJ)

    starts = [('e3-s1', E3[1]), ('e3-s3', E3[3]), ('e3-s4', E3[4]),
              ('e4-s0', E4[0]), ('e4-s1', E4[1]), ('e4-s2', E4[2])]
    fresh = [('f-e3-0', E3[0]), ('f-e3-2', E3[2]), ('f-e3-5', E3[5]),
             ('f-e4-3', E4[3]), ('f-e4-4', E4[4]), ('f-e4-5', E4[5])]
    objs = []
    objs += [('start-' + t, 'honest-start', p) for t, p in starts]
    objs += [('fresh-' + t, 'honest-fresh', p) for t, p in fresh]
    objs += [('orig-ce%d' % i, 'orig-ce', p) for i, p in
             enumerate(sorted(glob.glob(CE_ORIG)))]
    objs += [('orig-gr%d' % i, 'orig-gr', p) for i, p in
             enumerate(GR_ORIG)]
    p1_dir = [d for d in sorted(glob.glob(
        'research_outputs/uw/a8e-2*')) if 'smoke' not in d
        and 'ceJ_e3-s1.pt' in os.listdir(d)][0]
    objs += [('p1ce-' + t, 'p1-ce-joint',
              os.path.join(p1_dir, 'ceJ_%s.pt' % t))
             for t, _ in starts]
    a8r_dirs = sorted(glob.glob('research_outputs/uw/a8r-2*/regen'))
    if a8r_dirs:
        for p in sorted(glob.glob(
                os.path.join(a8r_dirs[-1], '*.pt'))):
            tag = os.path.basename(p)[:-3]
            objs.append((tag, 'grad-joint-regen', p))
    for d in sorted(glob.glob('research_outputs/uw/a8e-2*')):
        if 'smoke' in d:
            continue
        for p in sorted(glob.glob(os.path.join(d, 'ceJ_f-*.pt'))):
            tag = os.path.basename(p)[:-3].replace('ceJ_', '')
            objs.append(('p2a-ce-' + tag, 'p2a-ce-joint', p))

    print('%-22s %-18s %7s %7s %7s %7s'
          % ('tag', 'group', 's6', 'sJ', 'r_dual', 'dual_pass'), flush=True)
    rows = []
    for tag, grp, path in objs:
        try:
            sd = torch.load(path, map_location=device)
        except FileNotFoundError:
            continue
        s6, sj = scores(sd)
        rd = rho_dual(s6, sj)
        dp = s6 <= q20 and sj <= qJ
        rows.append({'tag': tag, 'group': grp, 's6': s6, 'sJ': sj,
                     'rho_dual': rd, 'dual_pass': dp})
        print('%-22s %-18s %7.3f %7.3f %7.3f %7s'
              % (tag, grp, s6, sj, rd, 'Y' if dp else 'N'), flush=True)

    hs = [r for r in rows if 'honest' in r['group']]
    af = [r for r in rows if 'joint' in r['group'] or 'orig' in
          r['group']]
    summary = {'honest_rho_mean': float(np.mean(
                   [r['rho_dual'] for r in hs])),
               'honest_rho_range': [float(min(
                   r['rho_dual'] for r in hs)), float(max(
                   r['rho_dual'] for r in hs))],
               'honest_dual_pass': sum(r['dual_pass'] for r in hs),
               'adaptive_rho_mean': float(np.mean(
                   [r['rho_dual'] for r in af])),
               'adaptive_dual_pass': sum(r['dual_pass'] for r in af),
               'qJ': float(qJ), 'q20': float(q20)}
    print(json.dumps(summary, indent=1), flush=True)
    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'rows': rows, 'summary': summary}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/dpe0-%s' % (
            time.strftime('%Y%m%dT%H%M%S'),)
    os.makedirs(cli.outdir, exist_ok=True)
    run(cli)


if __name__ == '__main__':
    main()

"""A8: joint (covariance-calibrated) acceptance rule vs the A6
marginal-band suite (prereg: docs/plans/2026-10-26-a8-joint-statistic-
prereg.md; zero new training).

s_J(theta) = (m - c)^T Sigmahat^{-1} (m - c), m = six-metric vector.
c = mean of fit (ew9F) metric vectors; Sigmahat = shrunk covariance
(1-rho)*S_fit + rho*diag(S_fit), rho = 0.5 frozen BEFORE any attack
readout; threshold = conformal quantile (alpha=0.20) of the cal
(p1aud+e2, 12 models) s_J scores under the frozen Sigmahat (cal does
not enter Sigmahat; clean split). Readouts: (i) held-out honest
acceptance (e3+e4, 12) under A8 vs A6 at matched alpha; (ii) encoded
finals (12, lfpr) rejection under A8 (A6 accepts 11/12, known);
(iii) E-SEQ stale trio scored descriptively.
Gates: G1 honest(A8) >= honest(A6) - 1; G2 reject >= 6/12 finals.
CLOSES = G1 and G2; COVERAGE-COST = G2 and not G1; NO-EFFECT = not G2.
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

CE_DIR = ('research_outputs/uw/lfpr-20261009T152330/ckpt/ce_*.pt')
GR = ['research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s1.pt',
      'research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s3.pt',
      'research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s4.pt',
      'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s0.pt',
      'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s1.pt',
      'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s2.pt']
STALE = [('base', 'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'),  # [1]
         ('ce', CE_DIR), ('gr', GR[0])]
ALPHA = 0.20


def vec(e):
    return np.array([e[m] for m in METRICS], dtype=float)


def run(cli, rho=0.5):
    device = torch.device('cuda:{}'.format(cli.gpu))
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[ALPHA]

    X_fit = np.stack([vec(e) for e in fit])            # (6, 6)
    c_j = X_fit.mean(0)
    S = np.cov(X_fit, rowvar=False)                    # rank-5
    Sh = (1 - rho) * S + rho * np.diag(np.diag(S))
    cond = float(np.linalg.cond(Sh))
    Sinv = np.linalg.inv(Sh)

    def sJ(e):
        d = vec(e) - c_j
        return float(d @ Sinv @ d)

    cal_scores = [sJ(e) for e in cal]
    qJ = conformal_q(cal_scores, ALPHA)
    print('rho=%.2f cond=%.1f qJ=%.4f (q6=%.4f)' % (rho, cond, qJ, q20),
          flush=True)

    ctx = setup_c10(device)

    def metrics_of(path):
        sd = torch.load(path, map_location=device)
        return ctx['eval_full'](sd)

    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'rho': rho, 'alpha': ALPHA, 'cond_Sigmahat': cond,
           'qJ': float(qJ), 'q20_marginal': float(q20), 'rows': []}

    def score_model(tag, group, e):
        s6 = sc(e)
        sj = sJ(e)
        row = {'tag': tag, 'group': group,
               's6': float(s6), 'pass6': bool(s6 <= q20),
               'sJ': float(sj), 'passJ': bool(sj <= qJ)}
        out['rows'].append(row)
        print(' %-8s %-10s s6=%7.3f (%s) sJ=%8.3f (%s)'
              % (tag, group, s6, 'P' if row['pass6'] else 'R',
                 sj, 'P' if row['passJ'] else 'R'), flush=True)
        return row

    if cli.smoke:
        test = test[:2]

    hon_rows = [score_model('h%d' % i, 'honest', e)
                for i, e in enumerate(test)]
    ce = sorted(glob.glob(CE_DIR))
    assert len(ce) == 6, ce
    fin_paths = [('ce%d' % i, p) for i, p in enumerate(ce)] \
        + [('gr%d' % i, p) for i, p in enumerate(GR)]
    fin_rows = [score_model(t, 'final', metrics_of(p))
                for t, p in (fin_paths[:1] if cli.smoke else fin_paths)]
    stale_rows = []
    if not cli.smoke:
        stale_rows = [
            score_model('stale-' + g, 'stale',
                        metrics_of(sorted(glob.glob(p))[
                            0 if g != 'base' else 1]))
            for g, p in STALE]

    hon6 = sum(r['pass6'] for r in hon_rows)
    honJ = sum(r['passJ'] for r in hon_rows)
    rejJ = sum(not r['passJ'] for r in fin_rows)
    g1 = honJ >= hon6 - 1
    g2 = rejJ >= (1 if cli.smoke else 6)
    verdict = ('CLOSES' if g1 and g2 else
               'COVERAGE-COST' if g2 else 'NO-EFFECT')
    out['gates'] = {'honest_pass_A6': int(hon6),
                    'honest_pass_A8': int(honJ),
                    'finals_rejected_A8': int(rejJ),
                    'n_finals': len(fin_rows),
                    'G1_equal_coverage': bool(g1),
                    'G2_gap_closes': bool(g2)}
    out['verdict'] = verdict
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', verdict,
          'honest %d/%d -> %d/%d, finals rejected %d/%d'
          % (hon6, len(hon_rows), honJ, len(hon_rows), rejJ,
             len(fin_rows)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--rho', default=0.5, type=float)
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/%s-%s' % (
            'smoke-a8joint' if cli.smoke else 'a8joint',
            time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(cli.outdir, exist_ok=True)
    cli and run(cli, cli.rho)


if __name__ == '__main__':
    main()

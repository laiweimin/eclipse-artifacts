"""D2: accuracy-budget audit across reference families (prereg addendum).

Mechanism under test: the exploitability budget is the accuracy component's
half-width in absolute accuracy points, B = q_hat * sigma_1, where sigma_1
is the MAD of the honest reference ensemble's m1. Predicts: families with a
small B deliver no signal, families with a large B do.

Families (honest reference models only, final models excluded):
  main-ft   : ew9F fit 6 (cat, k=30 ft refs)      + load_audit c/sigma
  k100-ft   : e3k100 ft_s* 24 (cat, k=100)
  frog/dog/airplane : e2mcls ckpt_*/ft_s* 24 each
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import setup_c10

FAM = {
    'main-ft-k30': ('research_outputs/uw/ew9F-20260927T001555/ckpt/'
                    'ft_k30_s*.pt', 5.245),
    'k100-ft': ('research_outputs/uw/e3k100-20261103T220400/ckpt/'
                'ft_s*.pt', None),
    'frog-ft': ('research_outputs/uw/e2mcls-20261103T220400/'
                'ckpt_frog/ft_s*.pt', 16.77),
    'dog-ft': ('research_outputs/uw/e2mcls-20261103T220400/'
               'ckpt_dog/ft_s*.pt', 4.46),
    'airplane-ft': ('research_outputs/uw/e2mcls-20261103T220400/'
                    'ckpt_airplane/ft_s*.pt', 4.48),
}
SIGNAL = {'main-ft-k30': 'yes (9/9)', 'k100-ft': 'no (0/6)',
          'frog-ft': 'yes (4/6)', 'dog-ft': 'partial (1/6)',
          'airplane-ft': 'no (0/6)'}


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    rows = []
    for name, (gl, q) in FAM.items():
        paths = sorted(glob.glob(gl))
        if not paths:
            print(' MISSING', name, gl, flush=True)
            continue
        v = []
        for p in paths:
            sd = torch.load(p, map_location=device)
            v.append(float(ctx['eval_full'](sd)['m1_acc']))
        c = float(np.median(v))
        sig = max(float(np.median(np.abs(np.array(v) - c))), 1e-9)
        qq = q if q is not None else float('nan')
        budget = qq * sig * 100.0 if q is not None else float('nan')
        rows.append({'family': name, 'n': len(v), 'c1': c, 'sig1': sig,
                     'q_hat': qq, 'budget_pp': budget,
                     'signal': SIGNAL[name],
                     'm1': [round(x, 4) for x in v]})
        print(' %-14s n=%2d c1=%.4f sig1=%.5f q=%5.2f B=%5.2fpp  signal=%s'
              % (name, len(v), c, sig, qq, budget, SIGNAL[name]),
              flush=True)
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump({'rows': rows}, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    cli.outdir = os.path.join('research_outputs', 'uw',
                              'budget-' + time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(cli.outdir, exist_ok=False)
    run(cli)


if __name__ == '__main__':
    main()

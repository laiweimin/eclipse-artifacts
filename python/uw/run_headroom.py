"""D1: utility headroom diagnostic (follow-up to path A, prereg addendum).

Hypothesis: signal delivery requires downward accuracy headroom below the
audit's m1 center. Compares, for the same three e4 starts, the normalized
m1 position (m1 - c1)/sigma1 under
  (a) the principal ft-reference audit (load_audit), and
  (b) the exact-retrain reference audit (E1 ckpts, 6 fit retrains).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import setup_c10
from uw.run_r1 import load_audit
from uw.run_ey import METRICS

E1_FIT = sorted(glob.glob(
    'research_outputs/uw/ertr-20261103T210519/ckpt/*fit*.pt')) or \
    sorted(glob.glob('research_outputs/uw/ertr-20261103T210519/ckpt/*.pt'))[:6]


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    _, _, _, c_ft, sig_ft, _, _ = load_audit()

    def m1(sd):
        return float(ctx['eval_full'](sd)['m1_acc'])

    e1 = [torch.load(p, map_location=device) for p in E1_FIT]
    v = [m1(sd) for sd in e1]
    c1_rt = float(np.median(v))
    s1_rt = max(float(np.median(np.abs(np.array(v) - c1_rt))), 1e-9)
    print(' E1 retrain m1: %s -> c=%.4f sig=%.4f'
          % ([round(x, 4) for x in v], c1_rt, s1_rt), flush=True)

    starts = sorted(glob.glob(
        'research_outputs/uw/ex3-20260929T101715/ckpt/e4_ft_s*.pt'))[:3]
    rows = []
    for p in starts:
        sd = torch.load(p, map_location=device)
        m = m1(sd)
        rows.append({'start': os.path.basename(p), 'm1': m,
                     'z_ft': (m - c_ft['m1_acc']) / sig_ft['m1_acc'],
                     'z_rt': (m - c1_rt) / s1_rt})
        print(' %s m1=%.4f  z_ft=%+.2f  z_rt=%+.2f'
              % (os.path.basename(p), m, rows[-1]['z_ft'],
                 rows[-1]['z_rt']), flush=True)

    out = {'c1_ft': c_ft['m1_acc'], 'sig1_ft': sig_ft['m1_acc'],
           'c1_rt': c1_rt, 'sig1_rt': s1_rt, 'rows': rows,
           'e1_fit_m1': [round(x, 5) for x in v]}
    # positive z means the model is ABOVE the center (better accuracy);
    # under a symmetric band the downward headroom is q_hat + z.
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    cli.outdir = os.path.join('research_outputs', 'uw',
                              'headroom-' + time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(cli.outdir, exist_ok=False)
    run(cli)


if __name__ == '__main__':
    main()

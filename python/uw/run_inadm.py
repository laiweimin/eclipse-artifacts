"""D4: why is theta0 inadmissible under the retrain-referenced audit?

Per-component decomposition of theta0's score against the E1 retrain
ensemble (c, sigma from rt_s47601..47606; q from rt_s47607..47612).
Also reports the ft-family (load_audit) position of theta0 for contrast.
"""
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import setup_c10
from uw.run_ey import METRICS
from uw.run_r1 import load_audit

FIT = sorted(glob.glob(
    'research_outputs/uw/ertr-20261103T210519/ckpt/rt_s4760[1-6].pt'))
CAL = sorted(glob.glob(
    'research_outputs/uw/ertr-20261103T210519/ckpt/rt_s4760[7-9].pt')) + \
    sorted(glob.glob(
        'research_outputs/uw/ertr-20261103T210519/ckpt/rt_s4761[0-2].pt'))
THETA0 = 'research_outputs/uw/ew3-20260925T113900/ckpt/theta0.pt'
START = 'research_outputs/uw/ex3-20260929T101715/ckpt/e4_ft_s46822.pt'


def main():
    dev = torch.device('cuda:1')
    ctx = setup_c10(dev)

    def mv(sd):
        m = ctx['eval_full'](sd)
        return {k: float(m[k]) for k in METRICS}

    fit = [mv(torch.load(p, map_location=dev)) for p in FIT]
    c = {k: float(np.median([e[k] for e in fit])) for k in METRICS}
    sg = {k: max(float(np.median(np.abs(
        np.array([e[k] for e in fit]) - c[k]))), 1e-12) for k in METRICS}
    cal = sorted(max(abs(e[k] - c[k]) / sg[k] for k in METRICS)
                 for e in [mv(torch.load(p, map_location=dev))
                           for p in CAL])
    q = float(cal[int(np.ceil(0.8 * len(cal))) - 1])

    _, _, _, c_ft, sg_ft, _, _ = load_audit()
    out = {'retrain_audit': {'c': c, 'sig': sg, 'q': q},
           'ft_audit': {'c': c_ft, 'sig': sg_ft}, 'models': {}}
    for tag, path in (('theta0', THETA0), ('ft_start_e4s1', START)):
        e = mv(torch.load(path, map_location=dev))
        z_rt = {k: (e[k] - c[k]) / sg[k] for k in METRICS}
        z_ft = {k: (e[k] - c_ft[k]) / sg_ft[k] for k in METRICS}
        out['models'][tag] = {'m': e, 'z_retrain': z_rt, 'z_ft': z_ft,
                              'score_retrain': max(abs(v) for v in z_rt.values()),
                              'score_ft': max(abs(v) for v in z_ft.values())}
        print(' %-14s score_rt=%.2f (q=%.2f)  score_ft=%.2f'
              % (tag, out['models'][tag]['score_retrain'], q,
                 out['models'][tag]['score_ft']), flush=True)
        for k in METRICS:
            print('   %-10s m=%.4f  z_rt=%+8.2f  z_ft=%+7.2f'
                  % (k, e[k], z_rt[k], z_ft[k]), flush=True)
    d = os.path.join('research_outputs', 'uw',
                     'inadmissible-' + time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(d, exist_ok=False)
    with open(os.path.join(d, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()

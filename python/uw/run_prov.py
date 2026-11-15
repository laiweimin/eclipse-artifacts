"""D3: provenance check -- theta0 vs ft vs retrain accuracy (path A follow-up)."""
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import setup_c10

CK = {
    'theta0 (ew3 stage-0)': ('research_outputs/uw/ew3-20260925T113900/'
                             'ckpt/theta0.pt', None),
    'ft start e4-s0': ('research_outputs/uw/ex3-20260929T101715/ckpt/'
                       'e4_ft_s46821.pt', None),
    'ft start e4-s1': ('research_outputs/uw/ex3-20260929T101715/ckpt/'
                       'e4_ft_s46822.pt', None),
    'ew9F ft ref s1': ('research_outputs/uw/ew9F-20260927T001555/ckpt/'
                       'ft_k30_s1.pt', None),
    'E1 retrain rt_s47601': ('research_outputs/uw/ertr-20261103T210519/'
                             'ckpt/rt_s47601.pt', None),
    'ew3 retrain rt_k30_s44101': ('research_outputs/uw/'
                                  'ew3-20260925T113900/ckpt/'
                                  'rt_k30_s44101.pt', None),
    'ew3 noise s44001 (full data)': ('research_outputs/uw/'
                                      'ew3-20260925T113900/ckpt/'
                                      'noise_s44001.pt', None),
}


def main():
    dev = torch.device('cuda:1')
    ctx = setup_c10(dev)
    out = {}
    for name, (p, _) in CK.items():
        if not os.path.exists(p):
            print(' MISSING', name, p, flush=True)
            continue
        sd = torch.load(p, map_location=dev)
        m = ctx['eval_full'](sd)
        out[name] = {'m1': round(float(m['m1_acc']), 4),
                     'm3': round(float(m['m3_Floss']), 4),
                     'path': p}
        print(' %-30s m1=%.4f m3=%.4f' % (name, m['m1_acc'],
                                          m['m3_Floss']), flush=True)
    d = os.path.join('research_outputs', 'uw',
                     'provenance-' + time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(d, exist_ok=False)
    with open(os.path.join(d, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()

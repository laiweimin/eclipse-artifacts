"""A3: aligned-instrument check (prereg 2026-11-12 path A).

Zero training. For the 12 primary finals and their 6 paired base starts,
measure neggrad AUC on two comparison populations (fresh 200 vs retained
3000, the audit m5 pool), raw and z-calibrated against the ew9F fit
ensemble. Decides POOL-MISMATCH-DOMINATES vs ALIGNED-SURVIVES.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import setup_c10
from uw.run_gm import auc as rank_auc
from uw.run_lfpr import channels

FIT_REFS = sorted(glob.glob(
    'research_outputs/uw/ew9F-20260927T001555/ckpt/ft_k30_s*.pt'))
CE = sorted(glob.glob(
    'research_outputs/uw/lfpr-20261009T152330/ckpt/ce_*.pt'))[:6]
GR = (sorted(glob.glob(
    'research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s*.pt'))
      + sorted(glob.glob(
          'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s*.pt')))
STARTS = (sorted(glob.glob(
    'research_outputs/uw/ex3-20260929T101715/ckpt/e3_ft_s*.pt'))
    + sorted(glob.glob(
        'research_outputs/uw/ex3-20260929T101715/ckpt/e4_ft_s*.pt')))


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    trset = CIFAR10(train=True, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    pool = ctx['pool']
    poolset = set(pool.tolist())
    fset = ctx['fset']
    fset_set = set(fset.tolist())
    mem_x, mem_y = ctx['xF'], ctx['yF']
    yfull_tr = np.asarray(trset.cifar10.targets)

    # fresh 200 (primary readout pool, run_ce seeds)
    cons = np.sort(np.random.RandomState(52801).choice(
        np.array([i for i in cat_te if i not in poolset]), 300,
        replace=False))
    fresh = np.sort(np.random.RandomState(52501).choice(
        np.setdiff1d(np.array([i for i in cat_te if i not in poolset]),
                     cons), 200, replace=False))
    # retained 3000 (ew3 sub seed; excludes pool)
    ret = np.sort(np.random.RandomState(52000).choice(
        np.setdiff1d(np.arange(len(yfull_tr)), pool), 3000,
        replace=False))

    def rec(src, idx):
        return (torch.tensor(np.stack(
                    [src.cifar10.data[i] / 255.0 for i in idx]))
                .permute(0, 3, 1, 2).float().to(device),
                torch.tensor([src.cifar10.targets[i] for i in idx])
                .long().to(device))

    fx, fy = rec(tset, fresh)
    rx, ry = rec(trset, ret)

    model = ctx['model']

    def ng(sd, x, y, chunk=100):
        """Per-record neggrad in chunks (avoids O(N^2) graph cost)."""
        outs = []
        for i in range(0, len(y), chunk):
            outs.append(channels(model, sd, x[i:i + chunk],
                                 y[i:i + chunk])['neggrad'])
        return np.concatenate(outs)

    # reference stats per record on both pools
    Rm = np.stack([ng(torch.load(p, map_location=device), mem_x, mem_y)
                   for p in FIT_REFS])
    Rf = np.stack([ng(torch.load(p, map_location=device), fx, fy)
                   for p in FIT_REFS])
    Rr = np.stack([ng(torch.load(p, map_location=device), rx, ry)
                   for p in FIT_REFS])
    mm_, sm_ = Rm.mean(0), Rm.std(0) + 1e-12
    print(' refs done', flush=True)
    mf, sf = Rf.mean(0), Rf.std(0) + 1e-12
    mr, sr = Rr.mean(0), Rr.std(0) + 1e-12

    def read(sd):
        z = ng(sd, mem_x, mem_y)
        non_f = ng(sd, fx, fy)
        non_r = ng(sd, rx, ry)
        out = {
            'fresh_raw': rank_auc(z, non_f),
            'ret_raw': rank_auc(z, non_r),
            'fresh_z': rank_auc((z - mm_) / sm_, (non_f - mf) / sf),
            'ret_z': rank_auc((z - mm_) / sm_, (non_r - mr) / sr),
        }
        return out

    rows = []
    for tag, ckpts in (('CE', CE), ('GR', GR)):
        for p in ckpts:
            rows.append({'tag': tag, 'ckpt': os.path.basename(p),
                         'final': read(torch.load(p, map_location=device))})
            print(' ', tag, os.path.basename(p),
                  {k: round(v, 3) for k, v in rows[-1]['final'].items()},
                  flush=True)
    bases = [read(torch.load(p, map_location=device)) for p in STARTS]
    base_mean = {k: float(np.mean([b[k] for b in bases])) for k in
                 ('fresh_raw', 'ret_raw', 'fresh_z', 'ret_z')}
    print(' base mean', {k: round(v, 3) for k, v in base_mean.items()},
          flush=True)

    # paired gains per final; bases are 6 starts, finals 12 -> pair by index
    # CE i and GR i derive from start i (e3-s1,s3,s4,e4-s0,s1,s2)
    gains = []
    for i, r in enumerate(rows):
        b = bases[i % len(bases)]
        gains.append({k: r['final'][k] - b[k] for k in b})
    med = {k: float(np.median([g[k] for g in gains])) for k in gains[0]}
    print(' median gains', {k: round(v, 3) for k, v in med.items()},
          flush=True)

    gf, gr = med['fresh_z'], med['ret_z']
    if gf <= 0:
        verdict = 'INCONCLUSIVE'
    elif gr <= gf / 3.0:
        verdict = 'POOL-MISMATCH-DOMINATES'
    elif gr >= 2.0 * gf / 3.0:
        verdict = 'ALIGNED-SURVIVES'
    else:
        verdict = 'MIXED'
    out = {'base_mean': base_mean, 'rows': rows, 'median_gains': med,
           'gains': gains, 'bases': bases, 'verdict': verdict}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT', verdict, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    tag = 'align-' + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(cli.outdir, exist_ok=False)
    run(cli)


if __name__ == '__main__':
    main()

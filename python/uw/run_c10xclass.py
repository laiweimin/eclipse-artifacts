"""C10: cross-class side-channel membership readout (zero training).

The audit monitors cat-class metrics (m2/m3/m4/m5/m6). Does the
CE encoding create membership-correlated signals in OTHER classes
that the audit completely ignores?

Method: take existing encoded finals + honest models, compute
z-scores using records from NON-cat classes (dog=5, frog=6, etc.)
as the evaluation set. If encoded models show elevated z on
non-cat records that correlate with cat-class F membership...
actually, this needs careful thought.

The right question: does the ENCODING (which optimizes cat-class
losses) create detectable changes in non-cat-class behavior that
distinguish encoded from honest models? This is a model-level
side channel, not a membership channel.

Simpler test: compute z-scores of encoded finals using dog-class
records (fresh non-members from dog class, no reference to F).
If the encoded models' z-scores on dog records are elevated vs
honest models, the encoding "leaks" through a channel the audit
doesn't monitor.
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
from uw.run_lfpr import channels, rec, zcal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    tag = 'c10xclass-' + time.strftime('%Y%m%dT%H%M%S')
    outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(outdir, exist_ok=False)
    device = torch.device('cuda:%d' % cli.gpu)

    ctx = setup_c10(device)
    model = ctx['model']
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False,
                   transform=_tf.Compose([_tf.ToTensor()]))
    targets = np.asarray(tset.cifar10.targets)

    # F records (cat class = 3)
    mem = (ctx['xF'], ctx['yF'])

    # Non-cat probe sets: dog(5), frog(6), ship(8), horse(7)
    probe_classes = {5: 'dog', 6: 'frog', 7: 'horse',
                     8: 'ship'}
    probe_sets = {}
    for cls, cname in probe_classes.items():
        idx = np.where(targets == cls)[0]
        # take 30 records as "probe members" (not actual members,
        # just probing for model-level side channel)
        sel = np.sort(np.random.RandomState(7700 + cls).choice(
            idx, 30, replace=False))
        probe_sets[cname] = rec(tset, sel, device)

    # Fresh non-members for FPR (also from each probe class)
    cat_te = np.where(targets == 3)[0]
    cons = np.sort(np.random.RandomState(52801).choice(cat_te, 300,
                                                       replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh_cat = np.sort(np.random.RandomState(52501).choice(
        rest, 200, replace=False))
    aux_cat = np.sort(np.random.RandomState(53304).choice(
        np.setdiff1d(rest, fresh_cat), 100, replace=False))
    fx_cat = rec(tset, fresh_cat, device)
    ax_cat = rec(tset, aux_cat, device)

    # For each probe class: fresh non-members from that class
    probe_fresh = {}
    for cls, cname in probe_classes.items():
        idx = np.where(targets == cls)[0]
        # exclude the 30 probe records
        remaining = np.setdiff1d(idx, sel)
        sel_fresh = np.sort(
            np.random.RandomState(7800 + cls).choice(
                remaining, 100, replace=False))
        probe_fresh[cname] = rec(tset, sel_fresh, device)

    # Reference ensemble (honest ft models)
    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-20260927T001555/ckpt/'
        'ft_k30_s*.pt'))
    fit_sds = [torch.load(p, map_location=device)
               for p in fit_paths]

    # Score each model on: F(cat), cat fresh, cat aux,
    # each probe class 30 + 100 fresh
    all_sets = {'mem': mem, 'cat_fresh': fx_cat,
                'cat_aux': ax_cat}
    for cname in probe_classes.values():
        all_sets['probe_' + cname] = probe_sets[cname]
        all_sets['fresh_' + cname] = probe_fresh[cname]

    fit_scores = [{nm: channels(model, sd, x, y)
                   for nm, (x, y) in all_sets.items()}
                  for sd in fit_sds]

    def zread_all(sd):
        s = {nm: channels(model, sd, x, y)
             for nm, (x, y) in all_sets.items()}
        out = {}
        for chn in ('negloss', 'neggrad'):
            for nm in all_sets:
                z = zcal(s[nm][chn],
                         [fs[nm][chn] for fs in fit_scores])
                out['%s|%s' % (nm, chn)] = {
                    'mean': float(np.mean(z)),
                    'max': float(np.max(z)),
                    'std': float(np.std(z))}
        return out

    honest_paths = fit_paths + sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/'
        'e3_ft_s*.pt'))[:6] + sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/'
            'e4_ft_s*.pt'))[:6]
    h1_paths = sorted(glob.glob(
        'research_outputs/uw/lfpr-*/ckpt/ce_*.pt'))[-6:] \
        + sorted(glob.glob(
            'research_outputs/uw/gate-g1b-20261007T102014/'
            'ckpt/fb_s*.pt')) \
        + sorted(glob.glob(
            'research_outputs/uw/gate-g1b-20261007T104042/'
            'ckpt/fb_s*.pt'))

    rows = []
    for label, paths in (('H0', honest_paths),
                         ('H1', h1_paths)):
        for i, pth in enumerate(paths):
            zr = zread_all(torch.load(pth, map_location=device))
            rows.append({'arm': label, 'i': i, 'zr': zr})
            # print key channels
            mem_z = zr.get('mem|neggrad', {}).get('mean', 0)
            dog_z = zr.get('probe_dog|neggrad',
                           {}).get('mean', 0)
            frog_z = zr.get('probe_frog|neggrad',
                            {}).get('mean', 0)
            print(' %s %d mem_z %.2f dog_z %.2f frog_z %.2f'
                  % (label, i, mem_z, dog_z, frog_z),
                  flush=True)

    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump({'generated':
                   time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'rows': rows}, fh, indent=1)

    # Summary: for each channel, compare H0 vs H1
    print('\\n=== CROSS-CHANNEL SIDE-LEAKAGE SUMMARY ===',
          flush=True)
    channels_list = [k for k in rows[0]['zr']
                     if 'neggrad' in k]
    for ch in sorted(channels_list):
        h0_vals = [r['zr'][ch]['mean'] for r in rows
                   if r['arm'] == 'H0']
        h1_vals = [r['zr'][ch]['mean'] for r in rows
                   if r['arm'] == 'H1']
        d = float(np.median(h1_vals)) - float(
            np.median(h0_vals))
        print(' %-25s H0med %+.3f H1med %+.3f diff %+.3f'
              % (ch, float(np.median(h0_vals)),
                 float(np.median(h1_vals)), d), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()

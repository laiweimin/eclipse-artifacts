"""C11: reference-free cross-class differential membership inference.

Key idea: the z-decoder needs 6 honest reference models. This new
readout needs ZERO references -- it uses the SAME model's dog-class
behavior as an internal baseline.

For each F record x_i:
  diff_i = score(x_i) - median_score(dog_probe)
where score = negloss or neggrad on the ENCODED model itself.

If diff_i is elevated for members (low cat loss due to encoding)
vs non-members (fresh cat records, no encoding effect), this is a
reference-free membership test.

The audit can't easily block this: it would need to monitor the
cross-class RELATIONSHIP (cat score relative to dog score), not
just individual class metrics.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from uw.run_ez import setup_c10
from uw.run_gm import auc as rank_auc
from uw.run_ey import K
from uw.run_ew3 import f_deep
from uw.run_lfpr import channels, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    tag = 'c11xclass-' + time.strftime('%Y%m%dT%H%M%S')
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
    cat_te = np.where(targets == 3)[0]
    cons = np.sort(np.random.RandomState(52801).choice(cat_te, 300,
                                                       replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    aux = np.sort(np.random.RandomState(53304).choice(
        np.setdiff1d(rest, fresh), 100, replace=False))
    mem = (ctx['xF'], ctx['yF'])
    fx = rec(tset, fresh, device)
    ax_ = rec(tset, aux, device)

    # dog probe set (30 records from dog class = 5)
    dog_idx = np.where(targets == 5)[0]
    dog_sel = np.sort(np.random.RandomState(7705).choice(
        dog_idx, 30, replace=False))
    dog = rec(tset, dog_sel, device)

    # horse probe (class 7) as second non-monitored baseline
    horse_idx = np.where(targets == 7)[0]
    horse_sel = np.sort(np.random.RandomState(7707).choice(
        horse_idx, 30, replace=False))
    horse = rec(tset, horse_sel, device)

    def score_diff(sd):
        """Reference-free cross-class differential for each F record.

        diff_i = score(x_i) - median(score(dog_probe))
        Also: diff_i_horse = score(x_i) - median(score(horse))
        """
        s_mem = channels(model, sd, mem[0], mem[1])
        s_dog = channels(model, sd, dog[0], dog[1])
        s_horse = channels(model, sd, horse[0], horse[1])
        s_fresh = channels(model, sd, fx[0], fx[1])
        s_aux = channels(model, sd, ax_[0], ax_[1])

        out = {}
        for chn in ('negloss', 'neggrad'):
            out[chn] = {}
        for chn in ('negloss', 'neggrad'):
            mem_s = s_mem[chn]
            dog_med = float(np.median(s_dog[chn]))
            horse_med = float(np.median(s_horse[chn]))
            fresh_s = s_fresh[chn]
            aux_s = s_aux[chn]

            # differential: member score - dog baseline
            diff_mem_dog = mem_s - dog_med
            diff_fresh_dog = fresh_s - dog_med
            diff_mem_horse = mem_s - horse_med
            diff_fresh_horse = fresh_s - horse_med

            # working point: threshold from aux differential
            diff_aux_dog = aux_s - dog_med
            diff_aux_horse = aux_s - horse_med
            tau_dog = float(np.quantile(diff_aux_dog, 0.95))
            tau_horse = float(np.quantile(
                diff_aux_horse, 0.95))

            # TPR and FPR
            tpr_dog = float(np.mean(diff_mem_dog > tau_dog))
            fpr_dog = float(np.mean(
                diff_fresh_dog > tau_dog))
            tpr_horse = float(np.mean(
                diff_mem_horse > tau_horse))
            fpr_horse = float(np.mean(
                diff_fresh_horse > tau_horse))

            # AUC: members vs fresh (differential)
            auc_dog = float(rank_auc(diff_mem_dog,
                                     diff_fresh_dog))
            auc_horse = float(rank_auc(diff_mem_horse,
                                       diff_fresh_horse))

            out[chn]['diff_dog'] = {
                'tpr5': tpr_dog, 'fpr5': fpr_dog,
                'auc': auc_dog}
            out[chn]['diff_horse'] = {
                'tpr5': tpr_horse, 'fpr5': fpr_horse,
                'auc': auc_horse}

            # also compute same-class baseline (no cross-class)
            # for comparison: mem vs fresh raw scores
            tau_same = float(np.quantile(aux_s, 0.95))
            out[chn]['same_class'] = {
                'tpr5': float(np.mean(mem_s > tau_same)),
                'fpr5': float(np.mean(fresh_s > tau_same)),
                'auc': float(rank_auc(mem_s, fresh_s))}
        return out

    honest_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-20260927T001555/ckpt/'
        'ft_k30_s*.pt')) + sorted(glob.glob(
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
    h1_names = (['CE:%d' % i for i in range(6)]
                + ['GR:%d' % i for i in range(6)])

    rows = []
    for label, paths, names in (('H0', honest_paths,
                                 ['HON:%d' % i
                                  for i in range(
                                      len(honest_paths))]),
                                ('H1', h1_paths, h1_names)):
        for pth, nm in zip(paths, names):
            sd = torch.load(pth, map_location=device)
            sr = score_diff(sd)
            rows.append({'arm': label, 'model': nm, 'sr': sr})
            print(' %s %s | dog: tpr %.2f fpr %.2f auc %.3f | '
                  'horse: tpr %.2f auc %.3f | same: tpr %.2f '
                  'auc %.3f'
                  % (label, nm,
                     sr['neggrad']['diff_dog']['tpr5'],
                     sr['neggrad']['diff_dog']['fpr5'],
                     sr['neggrad']['diff_dog']['auc'],
                     sr['neggrad']['diff_horse']['tpr5'],
                     sr['neggrad']['diff_horse']['auc'],
                     sr['neggrad']['same_class']['tpr5'],
                     sr['neggrad']['same_class']['auc']),
                  flush=True)

    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump({'generated':
                   time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'rows': rows}, fh, indent=1)

    # summary
    print('\\n=== C11 REFERENCE-FREE CROSS-CLASS DIFFERENTIAL ===',
          flush=True)
    for chn in ('negloss', 'neggrad'):
        for bt in ('diff_dog', 'diff_horse', 'same_class'):
            h0t = [r['sr'][chn][bt]['tpr5'] for r in rows
                   if r['arm'] == 'H0']
            h1t = [r['sr'][chn][bt]['tpr5'] for r in rows
                   if r['arm'] == 'H1']
            h0a = [r['sr'][chn][bt]['auc'] for r in rows
                   if r['arm'] == 'H0']
            h1a = [r['sr'][chn][bt]['auc'] for r in rows
                   if r['arm'] == 'H1']
            print(' %s|%s: H0 tpr %.2f auc %.3f | H1 tpr %.2f '
                  'auc %.3f | diff tpr %+.2f auc %+.3f'
                  % (chn, bt,
                     float(np.median(h0t)),
                     float(np.median(h0a)),
                     float(np.median(h1t)),
                     float(np.median(h1a)),
                     float(np.median(h1t))
                     - float(np.median(h0t)),
                     float(np.median(h1a))
                     - float(np.median(h0a))), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()

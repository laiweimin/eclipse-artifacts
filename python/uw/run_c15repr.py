"""C15: representation-level z-decoder (penultimate layer features).

All previous readouts used SCALAR scores (loss, margin, gradnorm).
This test uses the 512-dim penultimate activation VECTOR as the
per-record feature. The z-score becomes Mahalanobis distance from
the honest reference ensemble's per-record mean.

Hypothesis: representation-level features carry more membership
information than scalar projections, because the encoding's
gradient changes intermediate representations in directions that
output-level metrics don't capture.

Zero training: re-score existing encoded finals + honest models.
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
from uw.run_lfpr import rec, zcal


def penult_feat(model, sd, x, y):
    """Penultimate layer activations (before final FC)."""
    model.load_state_dict(sd)
    model.eval()

    def hook(module, input, output):
        hook.feat = output.detach()

    h = model.avgpool.register_forward_hook(hook)
    with torch.no_grad():
        model(x)
    h.remove()
    return hook.feat.flatten(1).cpu().numpy()  # (n, 512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    tag = 'c15repr-' + time.strftime('%Y%m%dT%H%M%S')
    outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(outdir, exist_ok=False)
    device = torch.device('cuda:%d' % cli.gpu)

    ctx = setup_c10(device)
    model = ctx['model']
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False,
                   transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(
        tset.cifar10.targets) == 3)[0]
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K)
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

    # reference ensemble
    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-20260927T001555/ckpt/'
        'ft_k30_s*.pt'))
    fit_sds = [torch.load(p, map_location=device)
               for p in fit_paths]

    # compute penultimate features for reference models
    ref_mem = np.stack([penult_feat(model, sd, mem[0], mem[1])
                        for sd in fit_sds])  # (6, 30, 512)
    ref_aux = np.stack([penult_feat(model, sd, ax_[0], ax_[1])
                        for sd in fit_sds])  # (6, 100, 512)
    ref_eval = np.stack([penult_feat(model, sd, fx[0], fx[1])
                         for sd in fit_sds])  # (6, 200, 512)

    def repr_zread(sd):
        """Mahalanobis-style z: distance from per-record ref mean
        in representation space, normalized by per-record ref std.

        For each record i:
          mu_i = mean over 6 refs (512-dim)
          d_i = ||feat_i - mu_i||_2
          z_i = d_i / median(d_ref_i)
        """
        fm = penult_feat(model, sd, mem[0], mem[1])  # (30, 512)
        fa = penult_feat(model, sd, ax_[0], ax_[1])
        fe = penult_feat(model, sd, fx[0], fx[1])

        # per-record reference stats
        mu_mem = ref_mem.mean(axis=0)  # (30, 512)
        mu_aux = ref_aux.mean(axis=0)  # (100, 512)
        mu_eval = ref_eval.mean(axis=0)  # (200, 512)

        # distance from reference mean
        d_mem = np.linalg.norm(fm - mu_mem, axis=1)  # (30,)
        d_aux = np.linalg.norm(fa - mu_aux, axis=1)  # (100,)
        d_eval = np.linalg.norm(fe - mu_eval, axis=1)  # (200,)

        # normalize by median reference distance (scale-free)
        ref_dists = []
        for j in range(6):
            for i in range(30):
                ref_dists.append(np.linalg.norm(
                    ref_mem[j, i] - mu_mem[i]))
        norm = float(np.median(ref_dists)) + 1e-12
        z_mem = d_mem / norm
        z_aux = d_aux / norm
        z_eval = d_eval / norm

        # working point
        tau = float(np.quantile(z_aux, 0.95))
        return {
            'tpr5': float(np.mean(z_mem > tau)),
            'fpr5': float(np.mean(z_eval > tau)),
            'auc': float(rank_auc(z_mem, z_eval)),
            'd_mem_med': float(np.median(d_mem)),
            'd_eval_med': float(np.median(d_eval))}

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
            zr = repr_zread(sd)
            rows.append({'arm': label, 'model': nm, 'zr': zr})
            print(' %s %s tpr5 %.2f fpr5 %.3f auc %.3f'
                  % (label, nm, zr['tpr5'], zr['fpr5'],
                     zr['auc']), flush=True)

    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump({'generated':
                   time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'rows': rows}, fh, indent=1)

    h0 = [r['zr'] for r in rows if r['arm'] == 'H0']
    h1 = [r['zr'] for r in rows if r['arm'] == 'H1']
    print('\\n=== REPRESENTATION Z DECODER ===', flush=True)
    print(' H0: tpr med %.2f auc med %.3f'
          % (float(np.median([r['tpr5'] for r in h0])),
             float(np.median([r['auc'] for r in h0]))),
          flush=True)
    print(' H1: tpr med %.2f auc med %.3f'
          % (float(np.median([r['tpr5'] for r in h1])),
             float(np.median([r['auc'] for r in h1]))),
          flush=True)
    print(' diff: tpr %+.2f auc %+.3f'
          % (float(np.median([r['tpr5'] for r in h1]))
             - float(np.median([r['tpr5'] for r in h0])),
             float(np.median([r['auc'] for r in h1]))
             - float(np.median([r['auc'] for r in h0]))),
          flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()

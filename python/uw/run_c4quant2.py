"""C4-v2: full-model per-channel INT4/INT2 quantization effect on
per-record z membership signals (extends c4quant which only
quantized nn.Linear; ResNet-18 is conv-dominated so that was ~0).

Protocol identical to c4quant prereg; amendment: quantize ALL
weight tensors (conv+fc) per-channel symmetric.
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


def quant_all(sd, bits):
    out = {}
    for k, v in sd.items():
        if 'weight' in k and v.dim() >= 2 and v.numel() > 100:
            flat = v.reshape(v.shape[0], -1)
            scale = flat.abs().max(dim=1, keepdim=True)[0] \
                / (2 ** (bits - 1) - 1)
            scale = scale.clamp(min=1e-10)
            q = torch.round(flat / scale).clamp(
                -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            out[k] = (q * scale).reshape(v.shape)
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    tag = 'c4quant2-' + time.strftime('%Y%m%dT%H%M%S')
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

    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-20260927T001555/ckpt/'
        'ft_k30_s*.pt'))
    fit_sds = [torch.load(p, map_location=device)
               for p in fit_paths]
    ev_sets = {'mem': mem, 'aux': ax_, 'eval': fx}
    fit_scores = [{nm: channels(model, sd, x, y)
                   for nm, (x, y) in ev_sets.items()}
                  for sd in fit_sds]

    def zread(sd):
        s = {nm: channels(model, sd, x, y)
             for nm, (x, y) in ev_sets.items()}
        out = {}
        for chn in ('negloss', 'neggrad'):
            zm = zcal(s['mem'][chn],
                      [fs['mem'][chn] for fs in fit_scores])
            za = zcal(s['aux'][chn],
                      [fs['aux'][chn] for fs in fit_scores])
            ze = zcal(s['eval'][chn],
                      [fs['eval'][chn] for fs in fit_scores])
            tau = float(np.quantile(za, 0.95))
            out['z_' + chn] = {
                'tpr5': float(np.mean(zm >= tau)),
                'fpr5': float(np.mean(ze >= tau)),
                'auc': float(rank_auc(zm, ze))}
        return out

    h1_paths = sorted(glob.glob(
        'research_outputs/uw/lfpr-*/ckpt/ce_*.pt'))[-6:] \
        + sorted(glob.glob(
            'research_outputs/uw/gate-g1b-20261007T102014/'
            'ckpt/fb_s*.pt')) \
        + sorted(glob.glob(
            'research_outputs/uw/gate-g1b-20261007T104042/'
            'ckpt/fb_s*.pt'))
    honest_paths = fit_paths + sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/'
        'e3_ft_s*.pt'))[:6] + sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/'
            'e4_ft_s*.pt'))[:6]

    rows = []
    for label, paths in (('H1', h1_paths),
                         ('H0', honest_paths)):
        for i, pth in enumerate(paths):
            sd = torch.load(pth, map_location=device)
            z_fp = zread(sd)
            sd4 = quant_all(sd, bits=4)
            z_4 = zread(sd4)
            sd2 = quant_all(sd, bits=2)
            z_2 = zread(sd2)
            d4 = z_4['z_neggrad']['tpr5'] \
                - z_fp['z_neggrad']['tpr5']
            d2 = z_2['z_neggrad']['tpr5'] \
                - z_fp['z_neggrad']['tpr5']
            rows.append({'arm': label, 'i': i, 'fp': z_fp,
                         'int4': z_4, 'int2': z_2})
            print(' %s %d tpr5 %.2f | int4 %.2f (d%+.2f) '
                  'fpr %.3f | int2 %.2f (d%+.2f) fpr %.3f'
                  % (label, i,
                     z_fp['z_neggrad']['tpr5'],
                     z_4['z_neggrad']['tpr5'], d4,
                     z_4['z_neggrad']['fpr5'],
                     z_2['z_neggrad']['tpr5'], d2,
                     z_2['z_neggrad']['fpr5']), flush=True)

    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump({'generated':
                   time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'rows': rows}, fh, indent=1)

    h1r = [r for r in rows if r['arm'] == 'H1']
    h0r = [r for r in rows if r['arm'] == 'H0']
    for q in ('int4', 'int2'):
        d_h1 = [r[q]['z_neggrad']['tpr5']
                - r['fp']['z_neggrad']['tpr5'] for r in h1r]
        d_h0 = [r[q]['z_neggrad']['tpr5']
                - r['fp']['z_neggrad']['tpr5'] for r in h0r]
        n_up = sum(1 for d in d_h1 if d >= 0.05)
        h0_med = float(np.median(d_h0))
        print(' %s: n_up=%d/%d H0med=%+.3f H1med=%+.3f'
              % (q, n_up, len(h1r), h0_med,
                 float(np.median(d_h1))), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()

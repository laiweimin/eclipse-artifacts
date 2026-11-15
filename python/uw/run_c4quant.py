"""C4-v1: quantization amplification of per-record membership
signals in audit-passing encoded finals (prereg card
docs/plans/2026-10-16-c4quant-prereg.md, frozen).

12 encoded finals (H1) + 12 honest (H0) -> dynamic INT8 quant
(nn.Linear) -> re-score z_negloss/z_neggrad per record -> paired
TPR@5% change. Gate: >=6/12 H1 finals improve >=+5pp paired,
H0 median change <=+3pp.
"""
import argparse
import copy
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
from uw.run_lfpr import channels, rec, zcal


def quantize_int8(sd, device):
    """Dynamic INT8 quantization of nn.Linear layers."""
    from tools.models import resnet18
    m = resnet18(num_classes=10)
    m.load_state_dict(sd)
    mq = torch.quantization.quantize_dynamic(
        m, {torch.nn.Linear}, dtype=torch.qint8)
    # extract dequantized weights back to state_dict form
    # (evaluation runs in fp with quantized weights baked in)
    sdq = {}
    for k, v in sd.items():
        sdq[k] = v
    # map quantized linear params back
    for name, mod in mq.named_modules():
        if isinstance(mod, torch.nn.quantized.Linear):
            # dequantize weights
            w = mod.weight().dequantize()
            sdq[name + '.weight'] = w
            if mod.bias() is not None:
                sdq[name + '.bias'] = mod.bias()
    return sdq


def quantize_int4(sd, device, bits=4):
    """Per-channel INT4 (simple symmetric round-to-nearest)."""
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
    tag = 'c4quant-' + time.strftime('%Y%m%dT%H%M%S')
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
    h1_names = (['CE:' + os.path.basename(p) for p in
                 sorted(glob.glob(
                     'research_outputs/uw/lfpr-*/ckpt/'
                     'ce_*.pt'))[-6:]]
                + ['GR:' + os.path.basename(p) for p in
                   sorted(glob.glob(
                       'research_outputs/uw/gate-g1b-'
                       '20261007T102014/ckpt/fb_s*.pt'))]
                + ['GR:' + os.path.basename(p) for p in
                   sorted(glob.glob(
                       'research_outputs/uw/gate-g1b-'
                       '20261007T104042/ckpt/fb_s*.pt'))])

    rows = []
    for label, paths, names in (('H1', h1_paths, h1_names),
                                ('H0', honest_paths,
                                 ['HON:%d' % i
                                  for i in range(
                                      len(honest_paths))])):
        for pth, nm in zip(paths, names):
            sd_fp = torch.load(pth, map_location=device)
            sd_q8 = quantize_int8(sd_fp, device)
            try:
                sd_q4 = quantize_int4(sd_fp, device)
                z4 = zread(sd_q4)
            except Exception:
                z4 = None
            z_fp = zread(sd_fp)
            z_q8 = zread(sd_q8)
            row = {'arm': label, 'model': nm,
                   'fp': z_fp, 'int8': z_q8, 'int4': z4}
            rows.append(row)
            d8 = z_q8['z_neggrad']['tpr5'] \
                - z_fp['z_neggrad']['tpr5']
            print(' %s %s tpr5(gr) %.2f->%.2f (d%+.2f) fpr '
                  '%.3f->%.3f'
                  % (label, nm, z_fp['z_neggrad']['tpr5'],
                     z_q8['z_neggrad']['tpr5'], d8,
                     z_fp['z_neggrad']['fpr5'],
                     z_q8['z_neggrad']['fpr5']), flush=True)

    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump({'generated':
                   time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'rows': rows}, fh, indent=1)

    h1_rows = [r for r in rows if r['arm'] == 'H1']
    h0_rows = [r for r in rows if r['arm'] == 'H0']
    d8_h1 = [r['int8']['z_neggrad']['tpr5']
             - r['fp']['z_neggrad']['tpr5'] for r in h1_rows]
    d8_h0 = [r['int8']['z_neggrad']['tpr5']
             - r['fp']['z_neggrad']['tpr5'] for r in h0_rows]
    n_up = sum(1 for d in d8_h1 if d >= 0.05)
    h0_med = float(np.median(d8_h0))
    gate = {'n_h1_up_5pp': n_up, 'n_h1': len(h1_rows),
            'h0_median_d': h0_med,
            'gate_pass': bool(n_up >= 6 and h0_med <= 0.03)}
    print('Q-GATE %s (n_up=%d/%d, H0 med %+.3f)'
          % (gate['gate_pass'], n_up, len(h1_rows), h0_med),
          flush=True)
    with open(os.path.join(outdir, 'gates.json'), 'w') as fh:
        json.dump(gate, fh, indent=1)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()

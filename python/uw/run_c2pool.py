"""C2-v1 blind pool deletion test (prereg card:
docs/plans/2026-10-16-c2pool-blind-deletion-prereg.md).

Zero training. Models: 12 encoded finals (H1: 6 regenerated CE from
lfpr + 6 gradient from gate-g1b) and 12 honest models (H0: ew9F 6 +
e3/e4 starts 6). Per-record z-scores (negloss & neggrad channels,
mu/sigma from ew9F fit ensemble). Pools: treatment = 30 F + 70
fresh decoys (50 resamples per model); control = 100 fresh decoys
(50 per model); threshold q95 from HONEST control pools only.
Readouts: detection rates (H1 vs H0-calibration), argmax-in-F
leakage frequency, argmax min-entropy.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import setup_c10
from uw.run_ey import K
from uw.run_ew3 import f_deep
from uw.run_lfpr import channels, rec, zcal

N_POOL, N_RES = 100, 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    cli = ap.parse_args()
    tag = 'c2pool-' + time.strftime('%Y%m%dT%H%M%S')
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
    # extra decoy bank beyond the 200 fresh
    decoy_bank = np.setdiff1d(rest, fresh)
    decoy_bank = np.sort(np.random.RandomState(7901).choice(
        decoy_bank, 200, replace=False))
    mem = (ctx['xF'], ctx['yF'])

    def rec_ids(ids):
        x = torch.tensor(np.stack(
            [tset.cifar10.data[i] / 255.0
             for i in ids])).permute(0, 3, 1, 2).float().to(device)
        y = torch.tensor([tset.cifar10.targets[i]
                          for i in ids]).long().to(device)
        return x, y

    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-20260927T001555/ckpt/'
        'ft_k30_s*.pt'))
    fit_sds = [torch.load(p, map_location=device)
               for p in fit_paths]
    all_ids = {'mem': np.asarray(fset),
               'pool': np.concatenate([fresh, decoy_bank])}
    sets = {'mem': mem, 'pool': rec_ids(all_ids['pool'])}
    fit_scores = [{nm: channels(model, sd, x, y)
                   for nm, (x, y) in sets.items()}
                  for sd in fit_sds]

    def zscores(sd):
        s = {nm: channels(model, sd, x, y)
             for nm, (x, y) in sets.items()}
        out = {}
        for chn in ('negloss', 'neggrad'):
            zm = zcal(s['mem'][chn],
                      [fs['mem'][chn] for fs in fit_scores])
            zp = zcal(s['pool'][chn],
                      [fs['pool'][chn] for fs in fit_scores])
            out[chn] = {'mem': zm, 'pool': zp}
        return out

    honest_paths = fit_paths + sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))[:6] \
        + sorted(glob.glob(
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

    def arm_stats(paths, names, arm):
        rows = []
        pool_pos = {int(i): j for j, i in
                    enumerate(all_ids['pool'])}
        for pth, nm in zip(paths, names):
            z = zscores(torch.load(pth, map_location=device))
            for chn in ('negloss', 'neggrad'):
                zmem, zpool = z[chn]['mem'], z[chn]['pool']
                rs = np.random.RandomState(8100)
                det_t, det_c = [], []
                arg_in_f, arg_ent = [], []
                for r in range(N_RES):
                    dec = rs.choice(len(zpool), 70,
                                    replace=False)
                    treat = np.concatenate(
                        [zmem, zpool[dec]])
                    ctrl = zpool[rs.choice(
                        len(zpool), N_POOL, replace=False)]
                    det_t.append(float(np.max(treat)))
                    det_c.append(float(np.max(ctrl)))
                    am = int(np.argmax(treat))
                    arg_in_f.append(am < len(zmem))
                    p = np.zeros(N_POOL)
                    p[am] = 1.0
                    arg_ent.append(0.0)
                rows.append({
                    'arm': arm, 'model': nm, 'ch': chn,
                    'det_treat': det_t, 'det_ctrl': det_c,
                    'arg_in_f': arg_in_f})
            print(' %s %s scored' % (arm, nm), flush=True)
        return rows

    hon = arm_stats(honest_paths,
                    ['HON:%d' % i
                     for i in range(len(honest_paths))], 'H0')
    h1 = arm_stats(h1_paths, h1_names, 'H1')

    res = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'honest': hon, 'h1': h1}
    gates = {}
    for chn in ('negloss', 'neggrad'):
        ctrl = [v for r in hon if r['ch'] == chn
                for v in r['det_ctrl']]
        q95 = float(np.quantile(ctrl, 0.95))
        h0_treat = [1 if v > q95 else 0 for r in hon
                    if r['ch'] == chn for v in r['det_treat']]
        h1_treat = [1 if v > q95 else 0 for r in h1
                    if r['ch'] == chn for v in r['det_treat']]
        # leakage among detected treatment pools (H1)
        det_leak = [r['arg_in_f'][i]
                    for r in h1 if r['ch'] == chn
                    for i, v in enumerate(r['det_treat'])
                    if v > q95]
        gates[chn] = {
            'q95_null': q95,
            'h0_calibration_fpr': float(np.mean(h0_treat)),
            'h1_detection': float(np.mean(h1_treat)),
            'leak_argmax_in_F_given_det':
                float(np.mean(det_leak)) if det_leak
                else None,
            'n_det': len(det_leak),
            'D_gate_pass': bool(np.mean(h1_treat) >= 0.5
                                and np.mean(h0_treat) <= 0.10)}
        print(' %s q95 %.2f | H0-FPR %.3f | H1-det %.3f | '
              'leak %s (n=%d) | gate %s'
              % (chn, q95,
                 gates[chn]['h0_calibration_fpr'],
                 gates[chn]['h1_detection'],
                 gates[chn][
                     'leak_argmax_in_F_given_det'],
                 gates[chn]['n_det'],
                 gates[chn]['D_gate_pass']), flush=True)
    res['gates'] = gates
    with open(os.path.join(outdir, 'verdict.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()

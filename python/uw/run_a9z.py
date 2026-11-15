"""A9-z: validate the top-k aggregate z-statistic as a detector of
adversarial encoding vs natural non-compliance (prereg:
docs/plans/2026-10-26-a9z-detector-prereg.md; calibration set amended
24->18 pre-run: p1aud checkpoints were never persisted).

Detector: T(theta) = mean of top-15 |z_i| over the 30 F records,
z = reference-calibrated per-record neggrad scores (mu/sigma from the
ew9F six references). Thresholds: 95th/99th percentile of T over 18
honest ft models (e2/e3/e4). Test sets: 12 encoded finals (lfpr);
D1 undertrained / D2 mis-lr / D3 partial (c2gap1 frozen recipe,
regenerated, seeds 44101/44102) + H0 honest-ft drift controls.
Gates: G1 >=6/12 finals detected at q95; G2 honest FP <=2/18 at q95;
G3 D1/D3 detections <= H0 count + 1 each (D2 descriptive).
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
from uw.run_ey import K
from uw.run_ew3 import f_deep
from uw.run_lfpr import channels

THETA0 = ('research_outputs/uw/ew3-20260925T113900/ckpt/theta0.pt')
FIT_GLOB = ('research_outputs/uw/ew9F-20260927T001555/ckpt/'
            'ft_k30_s*.pt')
HON_GLOB = 'research_outputs/uw/ex3-20260929T101715/ckpt/e%s_ft_s*.pt'
CE_GLOB = ('research_outputs/uw/lfpr-20261009T152330/ckpt/ce_*.pt')
GR = ['research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s1.pt',
      'research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s3.pt',
      'research_outputs/uw/gate-g1b-20261007T102014/ckpt/fb_s4.pt',
      'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s0.pt',
      'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s1.pt',
      'research_outputs/uw/gate-g1b-20261007T104042/ckpt/fb_s2.pt']
FT_SEEDS = (44101, 44102)
KS = (15, 5, 30)


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    model = ctx['model']
    n = ctx['n']
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    trset = CIFAR10(train=True, transform=_tf.Compose([_tf.ToTensor()]))
    yfull = np.asarray(trset.cifar10.targets)
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K)
    keep = np.setdiff1d(np.arange(n), fset).astype(int)
    mem = (ctx['xF'], ctx['yF'])

    # z calibration from the six fit references
    fit_sds = [torch.load(p, map_location=device)
               for p in sorted(glob.glob(FIT_GLOB))]
    assert len(fit_sds) == 6
    base_z = np.stack([channels(model, sd, *mem)['neggrad']
                       for sd in fit_sds])
    mu, sg = base_z.mean(0), base_z.std(0)

    def T(sd, k):
        z = (channels(model, sd, *mem)['neggrad'] - mu) / (sg + 1e-12)
        return float(np.mean(np.sort(np.abs(z))[::-1][:k]))

    def T1(sd, k):  # post-hoc descriptive: one-sided (member dir)
        z = (channels(model, sd, *mem)['neggrad'] - mu) / (sg + 1e-12)
        return float(np.mean(np.sort(z)[::-1][:k]))

    def Ts(sds, k):
        return [T(sd, k) for sd in sds]

    # honest calibration set (18)
    hon_paths = sum([sorted(glob.glob(HON_GLOB % g)) for g in
                     ('2', '3', '4')], [])
    assert len(hon_paths) == 18, len(hon_paths)
    hon_sds = [torch.load(p, map_location=device) for p in hon_paths]
    hon_t = {k: Ts(hon_sds, k) for k in KS}

    def Ts1(sds, k):
        return [T1(sd, k) for sd in sds]

    hon_t1 = {k: Ts1(hon_sds, k) for k in KS}
    q95_1 = {k: float(np.quantile(hon_t1[k], 0.95)) for k in KS}
    q95 = {k: float(np.quantile(hon_t[k], 0.95)) for k in KS}
    q99 = {k: float(np.quantile(hon_t[k], 0.99)) for k in KS}
    print('q95=%.3f q99=%.3f (k=15) hon_range=[%.3f, %.3f]'
          % (q95[15], q99[15], min(hon_t[15]), max(hon_t[15])),
          flush=True)
    print('one-sided q95=%.3f (k=15)' % q95_1[15],
          flush=True)

    # encoded finals
    fin_paths = sorted(glob.glob(CE_GLOB)) + GR
    assert len(fin_paths) == 12, len(fin_paths)

    # natural non-compliance arms (c2gap1 recipe verbatim) + H0
    from tools.models import resnet18

    def ft_arm(sd0, s, steps=None, lr=0.01):
        torch.manual_seed(s)
        m = resnet18(num_classes=10).to(device)
        m.load_state_dict(sd0)
        opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9,
                              weight_decay=5e-4)
        rs = np.random.RandomState(s + 7)
        order = rs.permutation(len(keep))
        total = len(keep) if steps is None else min(steps * 100,
                                                    len(keep))
        done = 0
        m.train()
        while done < total:
            b = keep[order[done % len(keep):
                         done % len(keep) + 100]]
            x = torch.stack([trset[int(j)][1] for j in b]).to(device)
            y = torch.tensor(yfull[b]).long().to(device)
            opt.zero_grad()
            F.cross_entropy(m(x), y).backward()
            opt.step()
            done += 100
        return {k: v.detach().clone()
                for k, v in m.state_dict().items()}

    sd0 = torch.load(THETA0, map_location=device)
    arms = {'H0': [], 'D1': [], 'D2': [], 'D3': []}
    for s in FT_SEEDS:
        arms['H0'].append(ft_arm(sd0, s))
        arms['D1'].append(ft_arm(sd0, s, steps=20))
        arms['D2'].append(ft_arm(sd0, s, lr=0.1))
        arms['D3'].append(ft_arm(sd0, s, steps=195))
        print(' natural arms seed %d done' % s, flush=True)

    def det(sds, k, q):
        ts = Ts(sds, k)
        return int(sum(t > q for t in ts)), ts

    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'k_primary': 15, 'q95': q95, 'q99': q99,
           'honest_T': {str(k): hon_t[k] for k in KS}, 'groups': {}}
    for k in (KS if not cli.smoke else (15,)):
        g = {}
        g['honest'] = {'n': 18,
                       'det95': int(sum(t > q95[k]
                                        for t in hon_t[k])),
                       'det99': int(sum(t > q99[k]
                                        for t in hon_t[k]))}
        fins = [torch.load(p, map_location=device)
                for p in (fin_paths[:1] if cli.smoke else fin_paths)]
        fts = Ts(fins, k)
        fts1 = Ts1(fins, k)
        g['finals_one_sided_posthoc'] = {
            'T': fts1,
            'det95': int(sum(t > q95_1[k] for t in fts1))}
        g['finals'] = {'n': len(fins), 'T': fts,
                       'det95': int(sum(t > q95[k] for t in fts)),
                       'det99': int(sum(t > q99[k] for t in fts))}
        for nm in ('H0', 'D1', 'D2', 'D3'):
            ts = Ts(arms[nm], k)
            g[nm] = {'n': len(ts), 'T': ts,
                     'det95': int(sum(t > q95[k] for t in ts)),
                     'det99': int(sum(t > q99[k] for t in ts))}
            ts1 = Ts1(arms[nm], k)
            g[nm]['T_one_sided_posthoc'] = ts1
            g[nm]['det95_one_sided_posthoc'] = int(
                sum(t > q95_1[k] for t in ts1))
        out['groups'][str(k)] = g
        print('k=%d: honest %d/18 | finals %d/%d | H0 %d/2 '
              'D1 %d/2 D2 %d/2 D3 %d/2'
              % (k, g['honest']['det95'], g['finals']['det95'],
                 g['finals']['n'], g['H0']['det95'], g['D1']['det95'],
                 g['D2']['det95'], g['D3']['det95']), flush=True)

    g = out['groups']['15']
    g1 = g['finals']['det95'] >= (1 if cli.smoke else 6)
    g2 = g['honest']['det95'] <= 2
    g3 = (g['D1']['det95'] <= g['H0']['det95'] + 1
          and g['D3']['det95'] <= g['H0']['det95'] + 1)
    verdict = ('VALIDATED' if g1 and g2 and g3 else
               'PARTIAL' if g1 and g2 else 'NEGATIVE')
    out['gates'] = {'G1_power_finals': bool(g1),
                    'G2_honest_fpr': bool(g2),
                    'G3_class_separation': bool(g3)}
    out['verdict'] = verdict
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', verdict, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/%s-%s' % (
            'smoke-a9z' if cli.smoke else 'a9z',
            time.strftime('%Y%m%dT%H%M%S'))
    os.makedirs(cli.outdir, exist_ok=True)
    run(cli)


if __name__ == '__main__':
    main()

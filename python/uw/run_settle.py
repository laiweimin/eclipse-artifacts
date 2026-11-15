"""Settling experiment (prereg 2026-11-13): ancestry vs allowance.

The cold review's confound: E1's exact-retrain arm encoded from retrain
starts (never saw F), so allowance and F-ancestry predict the same null.
theta0 is the missing cell: trained on all of D (saw F), retrain-quality
accuracy (0.8924) and *admissible* under the retrain-calibrated audit
(c_rt=0.8887, sigma_rt=0.0014, q_rt=5.638 -> z=+2.6). Encoding from theta0
under the retrain audit therefore holds ancestry fixed (saw F) while the
allowance is the retrain one (0.79 points).

Predictions, frozen before the run:
  * ancestry story: theta0 saw F, so a signal should be encodable -> CI-positive gain.
  * allowance story: the allowance is 0.79 points (below the 2.08 of the
    smallest tested kappa) -> gain <= ~0.06 with intervals crossing zero.
A paired control encodes from one retrain start under the same audit
(reproduces E1's null in this execution).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import apply_delta, setup_c10
from uw.run_ew3b import LR0, lr_update
from uw.run_gm import auc as rank_auc
from uw.run_ey import METRICS

FIT = sorted(glob.glob(
    'research_outputs/uw/ertr-20261103T210519/ckpt/rt_s4760[1-6].pt'))
CAL = sorted(glob.glob(
    'research_outputs/uw/ertr-20261103T210519/ckpt/rt_s4760[7-9].pt')) + \
    sorted(glob.glob(
        'research_outputs/uw/ertr-20261103T210519/ckpt/rt_s4761[0-2].pt'))
THETA0 = 'research_outputs/uw/ew3-20260925T113900/ckpt/theta0.pt'
CONTROL = 'research_outputs/uw/ertr-20261103T210519/ckpt/rt_s47613.pt'


def gen(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 4 if cli.smoke else 60
    ctx = setup_c10(device)
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    poolset = set(ctx['pool'].tolist())
    mem = (ctx['xF'], ctx['yF'])
    model = ctx['model']

    def mvec(sd):
        m = ctx['eval_full'](sd)
        return {k: float(m[k]) for k in METRICS}

    fit_mm = [mvec(torch.load(p, map_location=device)) for p in FIT]
    c = {k: float(np.median([e[k] for e in fit_mm])) for k in METRICS}
    sig = {k: max(float(np.median(np.abs(
        np.array([e[k] for e in fit_mm]) - c[k]))), 1e-12)
        for k in METRICS}

    def sc(sd):
        e = mvec(sd)
        return max(abs(e[k] - c[k]) / sig[k] for k in METRICS)

    cal_sc = sorted(sc(torch.load(p, map_location=device)) for p in CAL)
    q_rt = float(cal_sc[int(np.ceil(0.8 * len(cal_sc))) - 1])
    band = q_rt * sig['m1_acc'] * 100.0
    print(' retrain audit: c1=%.4f sig1=%.5f q=%.3f band=%.2fpp'
          % (c['m1_acc'], sig['m1_acc'], q_rt, band), flush=True)

    cons = np.sort(np.random.RandomState(52801).choice(
        np.array([i for i in cat_te if i not in poolset]), 300,
        replace=False))
    cx = (torch.tensor(np.stack([tset.cifar10.data[i] / 255.0
                                 for i in cons])).permute(0, 3, 1, 2)
          .float().to(device),
          torch.tensor([tset.cifar10.targets[i] for i in cons])
          .long().to(device))
    fresh = np.sort(np.random.RandomState(52501).choice(
        np.setdiff1d(np.array([i for i in cat_te if i not in poolset]),
                     cons), 200, replace=False))
    fx = (torch.tensor(np.stack([tset.cifar10.data[i] / 255.0
                                 for i in fresh])).permute(0, 3, 1, 2)
          .float().to(device),
          torch.tensor([tset.cifar10.targets[i] for i in fresh])
          .long().to(device))

    def ce_obj(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        return (torch.nn.functional.cross_entropy(model(mem[0]), mem[1])
                - 0.5 * torch.nn.functional.cross_entropy(model(nx), ny))

    def encode(sd):
        cur, lr = sd, LR0
        rs = np.random.RandomState(54600)
        stall = 0
        for _it in range(iters):
            idx = rs.choice(len(cx[1]), 60, replace=False)
            nx, ny = cx[0][idx], cx[1][idx]
            model.load_state_dict(cur)
            model.zero_grad()
            ce_obj(cur, nx, ny).backward()
            g = {nm: p.grad.detach().clone() for nm, p in
                 model.named_parameters() if p.grad is not None}
            acc = False
            for _bt in range(6):
                cand = apply_delta(cur, g, -lr)
                if sc(cand) <= q_rt:
                    model.load_state_dict(cur)
                    model.zero_grad()
                    if float(ce_obj(cand, nx, ny)) < \
                            float(ce_obj(cur, nx, ny)) - 1e-9:
                        cur, acc = cand, True
                        break
                lr *= 0.5
                if lr < 1e-9:
                    break
            lr = lr_update(lr, acc)
            stall = 0 if acc else stall + 1
            if stall > 12:
                break
        return cur

    def auc_nl(sd):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            zm = -torch.nn.functional.cross_entropy(
                model(mem[0]), mem[1], reduction='none').cpu().numpy()
            zf = -torch.nn.functional.cross_entropy(
                model(fx[0]), fx[1], reduction='none').cpu().numpy()
        return rank_auc(zm, zf)

    def ci(sd):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            zm = -torch.nn.functional.cross_entropy(
                model(mem[0]), mem[1], reduction='none').cpu().numpy()
            zf = -torch.nn.functional.cross_entropy(
                model(fx[0]), fx[1], reduction='none').cpu().numpy()
        rs = np.random.RandomState(54700)
        bs = [rank_auc(zm[rs.choice(len(zm), len(zm), True)], zf)
              for _ in range(2000 if not cli.smoke else 50)]
        return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    rows = []
    for tag, path in (('theta0 (saw F)', THETA0),
                      ('retrain (never saw F)', CONTROL)):
        sd = torch.load(path, map_location=device)
        s0 = sc(sd)
        base = auc_nl(sd)
        t0 = time.time()
        if s0 > q_rt:
            print(' %s INFEASIBLE s0=%.2f > q=%.2f' % (tag, s0, q_rt),
                  flush=True)
            rows.append({'arm': tag, 'feasible': False, 's0': s0,
                         'base': base})
            continue
        fF = encode(sd)
        gain = auc_nl(fF) - base
        lo, hi = ci(fF)
        mf = mvec(fF)
        rows.append({'arm': tag, 'feasible': True, 's0': s0,
                     'base': base, 'gain': gain,
                     'ci_lo': lo - base, 'ci_hi': hi - base,
                     's_fin': sc(fF), 'm1_fin': mf['m1_acc'],
                     'z1_fin': (mf['m1_acc'] - c['m1_acc'])
                     / sig['m1_acc'], 'secs': round(time.time() - t0)})
        print(' %-22s base %.3f gain %+.3f ci[%+.3f,%+.3f] '
              's_fin %.2f z1 %+.1f (%.0fs)'
              % (tag, base, gain, lo - base, hi - base, rows[-1]['s_fin'],
                 rows[-1]['z1_fin'], time.time() - t0), flush=True)

    out = {'audit': {'c1': c['m1_acc'], 'sig1': sig['m1_acc'],
                     'q': q_rt, 'band_pp': band}, 'rows': rows}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=1, type=int)
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'settle-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(cli.outdir, exist_ok=False)
    gen(cli)


if __name__ == '__main__':
    main()

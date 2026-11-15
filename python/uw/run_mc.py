"""UW-Gm M-C: adaptive CE attacks under the 7-metric audit.

Runs the canonical first-order CE construction against the
re-calibrated seventh-statistic audit on three representative
starts, reporting per-start absolute gain drops vs the 6-metric
baseline (paired: same starts, same construction data, same
evaluation).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import apply_delta, setup_c10
from uw.run_gm import auc as rank_auc
from uw.run_gm2 import trace_loss
from uw.run_r1 import load_audit
from uw.run_ew3b import LR0, lr_update
from uw.run_ey import K, conformal_q
from uw.run_ew3 import f_deep


def rec(src, idx, device):
    x = torch.tensor(np.stack([src.cifar10.data[i] / 255.0
                               for i in idx]))\
        .permute(0, 3, 1, 2).float().to(device)
    y = torch.tensor([src.cifar10.targets[i] for i in idx])\
        .long().to(device)
    return x, y


def gmean(ctx, sd, x, y):
    model = ctx['model']
    model.load_state_dict(sd)
    model.eval()
    xg = x.clone().requires_grad_(True)
    out = model(xg)
    loss = torch.nn.functional.cross_entropy(out, y, reduction='none')
    tot = 0.0
    for i in range(len(y)):
        g = torch.autograd.grad(loss[i], xg, retain_graph=True)[0][i]
        tot = tot + g.norm()
    return float(tot / len(y))


def run_mc(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 4 if cli.smoke else 60
    ctx = setup_c10(device)
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K)
    cons = np.sort(np.random.RandomState(52801).choice(cat_te, 300,
                                                       replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    na = np.sort(np.random.RandomState(53101).choice(cat_te, 100,
                                                     replace=False))
    mem = (ctx['xF'], ctx['yF'])
    fx = rec(tset, fresh, device)
    cx = rec(tset, cons, device)
    nx0 = rec(tset, na, device)
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]
    fit_paths = sorted(glob.glob(
        'research_outputs/uw/ew9F-2*/ckpt/ft_k30_s*.pt'))

    def dstat(sd):
        return gmean(ctx, sd, nx0[0], nx0[1]) - gmean(
            ctx, sd, mem[0], mem[1])
    d_fit = [dstat(torch.load(p, map_location=device))
             for p in fit_paths]
    d_med = float(np.median(d_fit))
    d_sig = max(float(np.median(np.abs(np.array(d_fit) - d_med))),
                1e-9)

    def sc7(sd):
        dpen = abs(dstat(sd) - d_med) / d_sig
        return max(sc(ctx['eval_full'](sd)), dpen)
    cal_paths = (sorted(glob.glob(
        'research_outputs/uw/r3p1-2*/ckpt/ft_aud_s*.pt'))
        + sorted(glob.glob(
            'research_outputs/uw/ex3-2*/ckpt/e2_ft_s*.pt')))
    q7 = conformal_q([sc7(torch.load(p, map_location=device))
                      for p in cal_paths], 0.20)
    print(' mc q7 %.3f' % q7, flush=True)
    model = ctx['model']

    def ce_obj(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        return (torch.nn.functional.cross_entropy(model(mem[0]),
                                                  mem[1])
                - 0.5 * torch.nn.functional.cross_entropy(model(nx),
                                                          ny))

    def gen(sd, score_fn, qlim):
        cur, lr = sd, LR0
        rs = np.random.RandomState(53800)
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
            for _bt in range(8):
                cand = apply_delta(cur, g, -lr)
                if score_fn(cand) <= qlim:
                    v = float(ce_obj(cand, nx, ny))
                    v0 = float(ce_obj(cur, nx, ny))
                    if v < v0 - 1e-9:
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

    def nl(sd):
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            zm = -torch.nn.functional.cross_entropy(
                model(mem[0]), mem[1], reduction='none').cpu().numpy()
            zf = -torch.nn.functional.cross_entropy(
                model(fx[0]), fx[1], reduction='none').cpu().numpy()
        return rank_auc(zm, zf)
    e4 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))
    rows = []
    for p in e4[:3]:
        sd = torch.load(p, map_location=device)
        base = nl(sd)
        f6 = gen(sd, lambda s_: sc(ctx['eval_full'](s_)), q20)
        a6 = nl(f6)
        f7 = gen(sd, sc7, q7)
        a7 = nl(f7)
        ok7 = bool(sc7(f7) <= q7)
        rows.append({'start': os.path.basename(p), 'base': base,
                     'gain6': a6 - base, 'gain7': a7 - base,
                     'pass7': ok7})
        print(' mc %s base %.3f g6 %.3f g7 %.3f pass %s'
              % (os.path.basename(p), base, a6 - base, a7 - base,
                 ok7), flush=True)
    drops = [r_['gain6'] - r_['gain7'] for r_ in rows]
    v = ('MC_COMPRESSED' if max(drops) > 0.05 else 'MC_NOT_COMPRESSED')
    out = {'q7': float(q7), 'rows': rows, 'drops': drops,
           'verdict': v}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT', v, '| drops', [round(d_, 3) for d_ in drops],
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'mc-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(cli.outdir, exist_ok=False)
    run_mc(cli)


if __name__ == '__main__':
    main()

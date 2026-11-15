"""A7 objective-redesign arm (prereg docs/plans/2026-11-09-a7-redesign-prereg.md).

Two redesigned objectives optimized under the recalibrated A7 audit
(same q7/d_med/d_sig construction as run_mc.py):
  R1: J_CE + 1.0 * (dstat(theta) - d_med)^2   (anti-statistic penalty)
  R2: J_G = E_F||grad_x l|| - 0.5 E_Nc||grad_x l||  (second-order objective)
Same three e4 starts, same record-set seeds, same label-blind eval (nl).
Gates frozen in the prereg card. CI via 2000 member resamples.
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
from uw.run_r1 import load_audit
from uw.run_ew3b import LR0, lr_update
from uw.run_ew3 import f_deep
from uw.run_ey import K, conformal_q


def rec(src, idx, device):
    x = torch.tensor(np.stack([src.cifar10.data[i] / 255.0
                               for i in idx]))\
        .permute(0, 3, 1, 2).float().to(device)
    y = torch.tensor([src.cifar10.targets[i] for i in idx])\
        .long().to(device)
    return x, y


def gmean_nodiff(ctx, sd, x, y):
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


def gmean_diff(model, x, y):
    """Differentiable E_i ||grad_x l_i|| (create_graph for higher-order)."""
    xg = x.clone().requires_grad_(True)
    out = model(xg)
    loss = torch.nn.functional.cross_entropy(out, y, reduction='none')
    gs = []
    for i in range(len(y)):
        g = torch.autograd.grad(loss[i], xg, retain_graph=True,
                                create_graph=True)[0][i]
        gs.append(g.norm())
    return torch.stack(gs).mean()


def run_mcr(cli):
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
        return gmean_nodiff(ctx, sd, nx0[0], nx0[1]) - gmean(
            ctx, sd, mem[0], mem[1]) if False else (
            gmean_nodiff(ctx, sd, nx0[0], nx0[1]) -
            gmean_nodiff(ctx, sd, mem[0], mem[1]))
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
    print(' mcr q7 %.3f' % q7, flush=True)
    model = ctx['model']

    def ce_obj(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        return (torch.nn.functional.cross_entropy(model(mem[0]),
                                                  mem[1])
                - 0.5 * torch.nn.functional.cross_entropy(model(nx),
                                                          ny))

    def j_r1(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        ce = (torch.nn.functional.cross_entropy(model(mem[0]),
                                                mem[1])
              - 0.5 * torch.nn.functional.cross_entropy(
                  model(nx), ny))
        d_diff = (gmean_diff(model, nx0[0], nx0[1])
                  - gmean_diff(model, mem[0], mem[1])) - d_med
        return ce + 1.0 * d_diff.pow(2)

    def j_r2(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        return gmean_diff(model, mem[0], mem[1]) \
            - 0.5 * gmean_diff(model, nx, ny)

    def gen(sd, jfn, qlim):
        cur, lr = sd, LR0
        rs = np.random.RandomState(53900)
        stall = 0
        for _it in range(iters):
            idx = rs.choice(len(cx[1]), 60, replace=False)
            nx, ny = cx[0][idx], cx[1][idx]
            model.load_state_dict(cur)
            model.zero_grad()
            jfn(cur, nx, ny).backward()
            g = {nm: p.grad.detach().clone() for nm, p in
                 model.named_parameters() if p.grad is not None}
            acc = False
            for _bt in range(8):
                cand = apply_delta(cur, g, -lr)
                if sc7(cand) <= qlim:
                    model.load_state_dict(cur)
                    model.zero_grad()
                    v = float(jfn(cand, nx, ny))
                    v0 = float(jfn(cur, nx, ny))
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

    def ci(sd):
        """2000 member-resample CI of the AUC gain vs base (same indices)."""
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            zm = -torch.nn.functional.cross_entropy(
                model(mem[0]), mem[1], reduction='none').cpu().numpy()
            zf = -torch.nn.functional.cross_entropy(
                model(fx[0]), fx[1], reduction='none').cpu().numpy()
        rs = np.random.RandomState(54000)
        bs = []
        for _ in range(2000 if not cli.smoke else 50):
            mi = rs.choice(len(zm), len(zm), replace=True)
            bs.append(rank_auc(zm[mi], zf))
        return float(np.percentile(bs, 2.5)), \
            float(np.percentile(bs, 97.5))

    e4 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))
    rows = []
    for p in e4[:3]:
        sd = torch.load(p, map_location=device)
        base = nl(sd)
        for arm, jfn in (('R1_penalty', j_r1), ('R2_secondorder', j_r2)):
            t0 = time.time()
            f7 = gen(sd, jfn, q7)
            a7 = nl(f7)
            lo, hi = ci(f7)
            ok7 = bool(sc7(f7) <= q7)
            rows.append({'start': os.path.basename(p), 'arm': arm,
                         'base': base, 'gain7': a7 - base,
                         'ci_lo': lo - base, 'ci_hi': hi - base,
                         'pass7': ok7,
                         'secs': round(time.time() - t0, 1)})
            print(' mcr %s %s base %.3f g7 %.3f ci[%.3f,%.3f] '
                  'pass %s (%.0fs)'
                  % (os.path.basename(p), arm, base, a7 - base,
                     lo - base, hi - base, ok7,
                     time.time() - t0), flush=True)
    ev = [r for r in rows if r['gain7'] >= 0.10
          and r['ci_lo'] > 0]
    rob = all(r['gain7'] <= 0.05 for r in rows)
    v = ('A7-EVASION' if ev else
         ('A7-COMPRESSION-ROBUST' if rob else 'A7-PARTIAL'))
    out = {'q7': float(q7), 'rows': rows, 'verdict': v}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT', v, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'mcr-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(cli.outdir, exist_ok=False)
    run_mcr(cli)


if __name__ == '__main__':
    main()

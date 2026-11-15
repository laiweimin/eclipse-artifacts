"""UW-Gm Gate G1: independent downstream inference (frozen prereg
docs/plans/2026-10-07-uwgm-gates-*.md, decisive zero-training gate).

Regenerate and freeze the three directed finals; hand them to a
label-blind inferrer that uses (i) fresh same-class non-members
never used in construction, (ii) a threshold fixed on a disjoint
auxiliary non-member set without any member labels, and (iii)
reports AUC / TPR@FPR10 / member rank-percentile against BOTH
never-trained and retained same-class candidates, alongside the
honest starts and theta0.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import setup_c10
from uw.run_gm import grad_norms, auc
from uw.run_gm2 import ascend_trace
from uw.run_r1 import RUN3, load_audit


def cat_records(device, src, idx):
    x = torch.tensor(np.stack([src.cifar10.data[i] / 255.0
                               for i in idx]))\
        .permute(0, 3, 1, 2).float().to(device)
    y = torch.tensor([src.cifar10.targets[i] for i in idx])\
        .long().to(device)
    return x, y


def run_g1(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 3 if cli.smoke else 60
    ctx = setup_c10(device)
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]

    def score_sd(sd):
        return sc(ctx['eval_full'](sd))
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    trset = CIFAR10(train=True, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    cat_tr = ctx['pool']                       # 500 cat train records
    from uw.run_ey import K
    from uw.run_ew3 import f_deep
    fset = f_deep(cat_tr, K)
    # fresh non-members / aux (never used in construction)
    r = np.random.RandomState(52501)
    fresh = np.sort(r.choice(cat_te, 30, replace=False))
    r2 = np.random.RandomState(52601)
    aux = np.sort(r2.choice(np.setdiff1d(cat_te, fresh), 30,
                            replace=False))
    # retained same-class candidates (trained, not deleted)
    r3 = np.random.RandomState(52701)
    ret = np.sort(r3.choice(np.setdiff1d(cat_tr, fset), 30,
                            replace=False))
    fx, fy = cat_records(device, tset, fresh)
    ax_, ay = cat_records(device, tset, aux)
    rx, ry = cat_records(device, trset, ret)
    mem_x, mem_y = ctx['xF'], ctx['yF']
    starts = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    ck = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)

    def normfeats(sd, x, y):
        return -grad_norms(ctx, sd, x, y)     # higher = member-like

    def labelblind(sd):
        """Inferrer: threshold from aux (label-blind); metrics vs
        fresh and vs retained."""
        fa = normfeats(sd, ax_, ay)
        tau = float(np.median(fa))
        fm = normfeats(sd, mem_x, mem_y)
        ff = normfeats(sd, fx, fy)
        fr = normfeats(sd, rx, ry)
        a_fresh = auc(fm, ff)
        a_ret = auc(fm, fr)
        thr_fpr = float(np.quantile(ff, 0.90))       # FPR 10% point
        tpr = float(np.mean(fm >= thr_fpr))
        thr_fpr2 = float(np.quantile(ff, 0.80))      # FPR 20%
        tpr2 = float(np.mean(fm >= thr_fpr2))
        rankpct = float(np.mean([(fm >= v).mean()
                                 for v in ff]))
        return {'auc_fresh': a_fresh, 'auc_retained': a_ret,
                'tpr@fpr10': tpr, 'tpr@fpr20': tpr2,
                'rank_pct': rankpct, 'tau_med_aux': tau}

    r0 = np.random.RandomState(52001)
    orig_nm = np.sort(r0.choice(cat_te, 30, replace=False))
    ox, oy = cat_records(device, tset, orig_nm)   # construction set
    rows = []
    for j in (1, 3, 4):
        sd = torch.load(starts[j], map_location=device)
        fin, _ = ascend_trace(ctx, sd, score_sd, q20, mem_x, mem_y,
                              ox, oy, iters)      # ORIGINAL N_c
        torch.save(fin, os.path.join(ck, 'fin_s{}.pt'.format(j)))
        for name, sdx in (('base', sd), ('directed', fin)):
            m = labelblind(sdx)
            m.update({'start': j, 'model': name})
            rows.append(m)
            print(' g1 s%d %-9s fresh %.3f ret %.3f tpr10 %.2f'
                  % (j, name, m['auc_fresh'], m['auc_retained'],
                     m['tpr@fpr10']), flush=True)
    sd0 = torch.load(os.path.join(RUN3, 'ckpt', 'theta0.pt'),
                     map_location=device)
    m0 = labelblind(sd0)
    m0.update({'start': 'theta0', 'model': 'theta0'})
    rows.append(m0)
    print(' g1 theta0 fresh %.3f ret %.3f'
          % (m0['auc_fresh'], m0['auc_retained']), flush=True)
    gains = []
    for j in (1, 3, 4):
        b = [r_['auc_fresh'] for r_ in rows
             if r_['start'] == j and r_['model'] == 'base'][0]
        d = [r_['auc_fresh'] for r_ in rows
             if r_['start'] == j and r_['model'] == 'directed'][0]
        gains.append(d - b)
    ok = sum(1 for g_ in gains if g_ >= 0.05)
    verdict = ('GATE1_PASS' if ok >= 2 else 'GATE1_FAIL')
    out = {'rows': rows, 'gains': gains, 'verdict': verdict}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT', verdict, '| gains', [round(g_, 3)
                                           for g_ in gains],
          flush=True)


def run_g1b(cli):
    """Gate G1b: construction N_c=300 (stochastic 60/step), label-
    blind evaluation on 200 fresh non-members (sigma~0.03)."""
    import torch
    device = torch.device('cuda:{}'.format(cli.gpu))
    iters = 3 if cli.smoke else 60
    ctx = setup_c10(device)
    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]

    def score_sd(sd):
        return sc(ctx['eval_full'](sd))
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    from uw.run_ez import apply_delta
    from uw.run_gm2 import trace_loss
    from uw.run_ew3b import LR0, lr_update
    from uw.run_ey import K
    from uw.run_ew3 import f_deep
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    trset = CIFAR10(train=True, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K)
    rng = np.random.RandomState(52801)
    cons = np.sort(rng.choice(cat_te, 300, replace=False))
    rest = np.setdiff1d(cat_te, cons)
    rng2 = np.random.RandomState(52501)
    fresh = np.sort(rng2.choice(rest, 200, replace=False))
    rest2 = np.setdiff1d(rest, fresh)
    aux = np.sort(np.random.RandomState(52901).choice(rest2, 100,
                                                       replace=False))
    r3 = np.random.RandomState(52701)
    ret = np.sort(r3.choice(np.setdiff1d(cat_tr, fset), 30,
                            replace=False))
    cx, cy = cat_records(device, tset, cons)
    fx, fy = cat_records(device, tset, fresh)
    ax_, ay = cat_records(device, tset, aux)
    rx, ry = cat_records(device, trset, ret)
    mem_x, mem_y = ctx['xF'], ctx['yF']
    model = ctx['model']

    def ascend_stoch(sd):
        cur, lr = sd, LR0
        rs = np.random.RandomState(52811)
        stall = 0
        for _it in range(iters):
            idx = rs.choice(len(cy), 60, replace=False)
            nx = cx[idx]
            ny = cy[idx]
            model.load_state_dict(cur)
            model.zero_grad()
            trace_loss(ctx, cur, mem_x, mem_y, nx, ny).backward()
            g = {nm: p.grad.detach().clone() for nm, p in
                 model.named_parameters() if p.grad is not None}
            acc = False
            for _bt in range(8):
                cand = apply_delta(cur, g, -lr)
                if score_sd(cand) <= q20:
                    v = float(trace_loss(ctx, cand, mem_x, mem_y,
                                         nx, ny))
                    if v < float(trace_loss(ctx, cur, mem_x, mem_y,
                                            nx, ny)) - 1e-9:
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

    def normfeats(sd, x, y):
        return -grad_norms(ctx, sd, x, y)

    def labelblind(sd):
        tau = float(np.median(normfeats(sd, ax_, ay)))
        fm = normfeats(sd, mem_x, mem_y)
        ff = normfeats(sd, fx, fy)
        fr = normfeats(sd, rx, ry)
        t10 = float(np.mean(fm >= np.quantile(ff, 0.90)))
        return {'auc_fresh200': auc(fm, ff), 'auc_ret': auc(fm, fr),
                'tpr@fpr10': t10, 'tau': tau}

    starts = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/{}_ft_s*.pt'.format(
            getattr(cli, 'pool', 'e3'))))
    if getattr(cli, 'pool', 'e3') == 'e4':
        starts = [p for p in starts
                  if score_sd(torch.load(p, map_location=device))
                  <= q20][:3]
    ck = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)
    rows = []
    for j in range(min(3, len(starts))):
        sd = torch.load(starts[j], map_location=device)
        fin = ascend_stoch(sd)
        torch.save(fin, os.path.join(ck, 'fb_s{}.pt'.format(j)))
        for name, sdx in (('base', sd), ('directed300', fin)):
            m = labelblind(sdx)
            m.update({'start': j, 'model': name})
            rows.append(m)
            print(' g1b s%d %-11s fresh %.3f ret %.3f tpr10 %.2f'
                  % (j, name, m['auc_fresh200'], m['auc_ret'],
                     m['tpr@fpr10']), flush=True)
    gains = []
    for j in (1, 3, 4):
        b = [r_['auc_fresh200'] for r_ in rows
             if r_['start'] == j and r_['model'] == 'base'][0]
        d = [r_['auc_fresh200'] for r_ in rows
             if r_['start'] == j and r_['model'] == 'directed300'][0]
        gains.append(d - b)
    verdict = ('G1B_PASS' if sum(1 for g_ in gains
                                 if g_ >= 0.05) >= 2 else 'G1B_FAIL')
    out = {'rows': rows, 'gains': gains, 'verdict': verdict}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT', verdict, '| gains', [round(g_, 3)
                                           for g_ in gains],
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='g1', choices=('g1', 'g1b'))
    ap.add_argument('--pool', default='e3')
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--smoke', action='store_true')
    cli = ap.parse_args()
    tag = ('smoke-' if cli.smoke else '') + 'gate-' + cli.arm + '-' \
        + time.strftime('%Y%m%dT%H%M%S')
    cli.outdir = os.path.join('research_outputs', 'uw', tag)
    os.makedirs(cli.outdir, exist_ok=False)
    print('Gate', cli.arm, '->', cli.outdir, flush=True)
    dict(g1=run_g1, g1b=run_g1b)[cli.arm](cli)


if __name__ == '__main__':
    main()

"""E5 arms A-C (prereg docs/plans/2026-11-04-e5-ablation-prereg.md):
encoder-core ablations on the FROZEN primary A6 audit.
  A: lambda contrastive weight {0.25, 0.5, 1.0, 2.0}
  B: Nc pool size {30, 100, 300, 1000} (resampled from the cons pool;
     1000 extends to the full class test-side)
  C: iteration budget {15, 30, 60, 120}
Starts: e3-s1 / e3-s3 / e4-s0. Readout: AUC(ng) base->final + CI.
Gates frozen in the card.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import setup_c10, apply_delta
from uw.run_ey import K
from uw.run_ew3 import f_deep
from uw.run_ew3b import LR0, lr_update
from uw.run_lfpr import channels
from uw.run_r1 import load_audit
from uw.run_gm import auc as rank_auc

LAMBDAS = (0.25, 0.5, 1.0, 2.0)
NCS = (30, 100, 300, 800)  # 800 = test-side ceiling after cons+fresh
ITERS = (15, 30, 60, 120)
STARTS = (('e3-s1', 'e3', 1), ('e3-s3', 'e3', 3), ('e4-s0', 'e4', 0))


def rank_ci(fm, bm, fe, be, n=2000, seed=53900):
    rs = np.random.RandomState(seed)
    base, fin = rank_auc(bm, be), rank_auc(fm, fe)
    diffs = [rank_auc(fm[rs.choice(len(fm), len(fm), replace=True)],
                      fe) - rank_auc(bm[rs.choice(len(fm), len(fm),
                                                  replace=True)], be)
            for _ in range(n)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return base, fin, float(lo), float(hi)


def ce_gen_lam(model, mem, cx, score_sd, q20, sd0, n_iter, lam):
    """ce_gen verbatim, lambda and iteration count parameterized."""
    def ce_obj(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        return (torch.nn.functional.cross_entropy(model(mem[0]),
                                                  mem[1])
                - lam * torch.nn.functional.cross_entropy(model(nx), ny))

    cur, lr = sd0, LR0
    rs = np.random.RandomState(53600)
    stall = 0
    bsz = min(60, len(cx[1]))
    for _it in range(n_iter):
        idx = rs.choice(len(cx[1]), bsz, replace=False)
        nx, ny = cx[0][idx], cx[1][idx]
        model.load_state_dict(cur)
        model.zero_grad()
        ce_obj(cur, nx, ny).backward()
        g = {nm: p.grad.detach().clone() for nm, p in
             model.named_parameters() if p.grad is not None}
        acc = False
        for _bt in range(8):
            cand = apply_delta(cur, g, -lr)
            if score_sd(cand) <= q20:
                if float(ce_obj(cand, nx, ny)) < float(
                        ce_obj(cur, nx, ny)) - 1e-9:
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


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    ctx = setup_c10(device)
    model = ctx['model']
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    cat_tr = ctx['pool']
    fset = f_deep(cat_tr, K)
    cons_all = np.sort(np.random.RandomState(52801).choice(
        cat_te, 300, replace=False))
    rest = np.setdiff1d(cat_te, cons_all)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    mem = (ctx['xF'], ctx['yF'])
    from uw.run_mb import rec
    fx = rec(tset, fresh, device)
    cx300 = rec(tset, cons_all, device)

    fit, cal, test, c, sig, sc, q = load_audit()
    q20 = q[0.20]

    def score_sd(sd):
        return sc(ctx['eval_full'](sd))

    e3 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e3_ft_s*.pt'))
    e4 = sorted(glob.glob(
        'research_outputs/uw/ex3-2*/ckpt/e4_ft_s*.pt'))
    start_sds = {'e3-s1': e3[1], 'e3-s3': e3[3], 'e4-s0': e4[0]}

    def nc_pool(n):
        if n <= 300:
            # deterministic subsample of the frozen cons pool
            idx = np.sort(np.random.RandomState(53700).choice(
                300, n, replace=False))
            return (cx300[0][idx], cx300[1][idx])
        # extend with the rest of the class test side
        extra = np.setdiff1d(cat_te, cons_all)
        # exclude fresh eval records from construction
        extra = np.setdiff1d(extra, fresh)
        add = np.sort(np.random.RandomState(53701).choice(
            extra, n - 300, replace=False))
        ax2 = rec(tset, add, device)
        return (torch.cat([cx300[0], ax2[0]]),
                torch.cat([cx300[1], ax2[1]]))

    def one_run(tag, sd0_path, cx, n_iter, lam):
        sd0 = torch.load(sd0_path, map_location=device)
        bm = channels(model, sd0, *mem)['neggrad']
        fe = channels(model, sd0, fx[0], fx[1])['neggrad']
        fin = ce_gen_lam(model, mem, cx, score_sd, q20, sd0,
                         n_iter, lam)
        fm = channels(model, fin, *mem)['neggrad']
        ch_fin_f = channels(model, fin, fx[0], fx[1])
        base_auc, fin_auc, lo, hi = rank_ci(
            fm, bm, ch_fin_f['neggrad'], fe)
        return {'auc_base': base_auc, 'auc_fin': fin_auc,
                'ci': [lo, hi], 'ci_positive': bool(lo > 0)}

    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'q20': q20, 'A_lambda': {}, 'B_nc': {}, 'C_iters': {}}

    print('--- arm A: lambda ---', flush=True)
    for lam in LAMBDAS:
        out['A_lambda'][str(lam)] = []
        for tag, p in start_sds.items():
            r = one_run(tag, p, cx300, 60, lam)
            r['start'] = tag
            out['A_lambda'][str(lam)].append(r)
            g = r['auc_fin'] - r['auc_base']
            print('lam=%.2f %s gain %+.3f CI[%+.3f,%+.3f]'
                  % (lam, tag, g, r['ci'][0], r['ci'][1]), flush=True)

    print('--- arm B: Nc size ---', flush=True)
    for n in NCS:
        cx = nc_pool(n)
        out['B_nc'][str(n)] = []
        for tag, p in start_sds.items():
            r = one_run(tag, p, cx, 60, 0.5)
            r['start'] = tag
            out['B_nc'][str(n)].append(r)
            g = r['auc_fin'] - r['auc_base']
            print('Nc=%d %s gain %+.3f CI[%+.3f,%+.3f]'
                  % (n, tag, g, r['ci'][0], r['ci'][1]), flush=True)

    print('--- arm C: iterations ---', flush=True)
    for n_it in ITERS:
        out['C_iters'][str(n_it)] = []
        for tag, p in start_sds.items():
            r = one_run(tag, p, cx300, n_it, 0.5)
            r['start'] = tag
            out['C_iters'][str(n_it)].append(r)
            g = r['auc_fin'] - r['auc_base']
            print('iters=%d %s gain %+.3f CI[%+.3f,%+.3f]'
                  % (n_it, tag, g, r['ci'][0], r['ci'][1]), flush=True)

    def mean_gain(d, k):
        return float(np.mean([r['auc_fin'] - r['auc_base']
                              for r in d[k]]))

    lam_g = {k: mean_gain(out['A_lambda'], k) for k in
             out['A_lambda']}
    nc_g = {k: mean_gain(out['B_nc'], k) for k in out['B_nc']}
    it_g = {k: mean_gain(out['C_iters'], k) for k in out['C_iters']}
    out['mean_gains'] = {'A': lam_g, 'B': nc_g, 'C': it_g}
    lam_ratio = max(lam_g.values()) / max(min(lam_g.values()), 1e-9)
    out['A_verdict'] = ('LAMBDA-FLAT' if lam_ratio < 2
                        else 'LAMBDA-SENSITIVE')
    nc_ref = nc_g['300']
    out['B_verdict'] = ('NC-ROBUST' if any(
        nc_g[k] >= 0.9 * nc_ref for k in ('30', '100'))
        else 'NC-COLLAPSE' if all(
        nc_g[k] < 0.5 * nc_ref for k in ('30', '100'))
        else 'NC-GRADIENT')
    out['C_verdict'] = ('ITER-SATURATED'
                        if abs(it_g['30'] - it_g['60']) < 0.03
                        else 'ITER-HUNGRY')
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('A:', out['A_verdict'], '| B:', out['B_verdict'],
          '| C:', out['C_verdict'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/abl-%s' % (
            time.strftime('%Y%m%dT%H%M%S'),)
    assert not os.path.exists(cli.outdir), cli.outdir
    os.makedirs(cli.outdir)
    run(cli)


if __name__ == '__main__':
    main()

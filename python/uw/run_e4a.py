"""E4a (prereg docs/plans/2026-11-04-e4-closing-batch-prereg.md, arm A):
GR (double-backward) encoding under exact-retrain references.
Reuses E1's retrain family verbatim (ertr-20261103T210519/ckpt);
audit recomputed bit-identically via the run_ertr code path;
GR objective/optimizer from the run_probe.gen_gr_probe protocol
(60 iters, same LR/backtracking/stall rules), acceptance check =
s6-RT <= q_RT. Gates: RT-GR-SURVIVES (>=2/6 accepted & CI+) /
PARTIAL (1) / NEGATIVE (0).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from uw.run_ez import setup_c10
from uw.run_ey import K, METRICS, conformal_q
from uw.run_ew3 import f_deep
from uw.run_ew3b import LR0, lr_update
from uw.run_lfpr import channels
from uw.run_gm import auc as rank_auc

SRC = 'research_outputs/uw/ertr-20261103T210519/ckpt'
FIT = [47601 + i for i in range(6)]
CAL = [47607 + i for i in range(6)]
STARTS = [(47613, 'new'), (47614, 'new'), (47615, 'new'),
          (44101, 'ew3'), (44102, 'ew3'), (44103, 'ew3')]
OLD_GLOB = ('research_outputs/uw/ew3-20260925T113900/ckpt/'
            'rt_k30_s*.pt')


def rank_ci(fm, bm, fe, be, n=2000, seed=53900):
    rs = np.random.RandomState(seed)
    base, fin = rank_auc(bm, be), rank_auc(fm, fe)
    diffs = [rank_auc(fm[rs.choice(len(fm), len(fm), replace=True)],
                      fe) - rank_auc(bm[rs.choice(len(fm), len(fm),
                                                  replace=True)], be)
            for _ in range(n)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return base, fin, float(lo), float(hi)


def gen_gr(model, mem, cx, score_fn, q, sd0, iters=60):
    """GR encoding, run_probe.gen_gr_probe protocol."""
    def loss(sd, nx, ny):
        model.load_state_dict(sd)
        model.eval()
        xg = mem[0].clone().requires_grad_(True)
        xg2 = nx.clone().requires_grad_(True)
        lf = torch.nn.functional.cross_entropy(
            model(xg), mem[1], reduction='none')
        ln = torch.nn.functional.cross_entropy(
            model(xg2), ny, reduction='none')
        gf = torch.autograd.grad(lf.sum(), xg,
                                 create_graph=True)[0].flatten(1).norm(1)
        gn = torch.autograd.grad(ln.sum(), xg2,
                                 create_graph=True)[0].flatten(1).norm(1)
        return (gf - 0.5 * gn).sum()

    cur, lr = sd0, LR0
    rs = np.random.RandomState(53600)
    stall, nacc = 0, 0
    for itn in range(iters):
        idx = rs.choice(len(cx[1]), 60, replace=False)
        nx, ny = cx[0][idx], cx[1][idx]
        model.load_state_dict(cur)
        model.zero_grad()
        loss(cur, nx, ny).backward()
        g = {nm: p.grad.detach().clone() for nm, p in
             model.named_parameters() if p.grad is not None}
        acc = False
        for _bt in range(8):
            cand = {nm: (p.detach() + lr * g[nm]) if nm in g
                    else p for nm, p in cur.items()}
            if score_fn(cand) <= q:
                if float(loss(cand, nx, ny)) > float(loss(cur, nx, ny)) \
                        + 1e-9:
                    cur, acc = cand, True
                    nacc += 1
                    break
            lr *= 0.5
            if lr < 1e-9:
                break
        lr = lr_update(lr, acc)
        stall = 0 if acc else stall + 1
        if stall > 12:
            break
        if nacc == 0 and itn >= 15:
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
    cons = np.sort(np.random.RandomState(52801).choice(cat_te, 300,
                                                       replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    mem = (ctx['xF'], ctx['yF'])
    from uw.run_mb import rec
    fx = rec(tset, fresh, device)
    cx = rec(tset, cons, device)

    def load(s):
        return torch.load(os.path.join(SRC, 'rt_s%d.pt' % s),
                          map_location=device)

    fit_mm = [ctx['eval_full'](load(s)) for s in FIT]
    c = {m: float(np.median([e[m] for e in fit_mm])) for m in METRICS}
    sig = {m: float(np.median(np.abs(
        np.array([e[m] for e in fit_mm]) - c[m]))) for m in METRICS}

    def sc(sd):
        e = ctx['eval_full'](sd)
        return max(abs(e[m] - c[m]) / max(sig[m], 1e-12)
                   for m in METRICS)

    q = conformal_q([sc(load(s)) for s in CAL], 0.20)
    print('q_RT recomputed = %.3f (E1 recorded 5.638)' % q, flush=True)
    assert abs(q - 5.638) < 0.01, 'audit reproduction mismatch'

    old = sorted(glob.glob(OLD_GLOB))
    assert len(old) == 3, old
    old_by_seed = {44101 + i: old[i] for i in range(3)}
    start_sds = []
    for s, src in STARTS:
        p = (os.path.join(SRC, 'rt_s%d.pt' % s) if src == 'new'
             else old_by_seed[s])
        start_sds.append((s, src, torch.load(p, map_location=device)))

    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'q_rt': q, 'rows': []}
    for s, src, sd0 in start_sds:
        s0 = float(sc(sd0))
        bm = channels(model, sd0, *mem)['neggrad']
        fe = channels(model, sd0, fx[0], fx[1])['neggrad']
        fin = gen_gr(model, mem, cx, sc, q, sd0)
        sf = float(sc(fin))
        fm = channels(model, fin, *mem)['neggrad']
        ch_fin_f = channels(model, fin, fx[0], fx[1])
        base_auc, fin_auc, lo, hi = rank_ci(
            fm, bm, ch_fin_f['neggrad'], fe)
        torch.save(fin, os.path.join(cli.outdir,
                                     'gr_s%d.pt' % s))
        row = {'start_seed': s, 'start_src': src,
               's6_start': s0, 's6_fin': sf,
               'pass_fin': bool(sf <= q),
               'auc_base': base_auc, 'auc_fin': fin_auc,
               'ci': [lo, hi], 'ci_positive': bool(lo > 0)}
        out['rows'].append(row)
        print('GR %d(%s): s6 %.3f->%.3f pass=%s auc %.3f->%.3f '
              'CI[%.3f,%.3f]'
              % (s, src, s0, sf, row['pass_fin'], base_auc, fin_auc,
                 lo, hi), flush=True)

    surv = sum(1 for r in out['rows']
               if r['pass_fin'] and r['ci_positive'])
    out['n_survive'] = surv
    out['verdict'] = ('RT-GR-SURVIVES' if surv >= 2
                      else 'RT-GR-PARTIAL' if surv == 1
                      else 'RT-GR-NEGATIVE')
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', out['verdict'], '(%d/6)' % surv, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=0, type=int)
    ap.add_argument('--outdir', default='')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/ertr-gr-%s' % (
            time.strftime('%Y%m%dT%H%M%S'),)
    assert not os.path.exists(cli.outdir), cli.outdir
    os.makedirs(cli.outdir)
    run(cli)


if __name__ == '__main__':
    main()

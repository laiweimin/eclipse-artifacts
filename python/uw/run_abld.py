"""E5 arm D (prereg docs/plans/2026-11-04-e5-ablation-prereg.md):
F-selection-rule ablation. F_rand = 30 records drawn uniformly from
the cat pool (RandomState 47950) instead of f_deep's
distinctiveness selection. Full fresh pipeline: 24 one-epoch ft
(seeds 47901-24) -> A6-rand audit (6/12/6) -> ce_gen x 3 starts.
Gate: RANDF-REPLICATED (>=2/3 CI-positive) / WEAK (1) / NOT (0).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import uw.run_ez as ez
from uw.run_ey import K, METRICS, conformal_q
from uw.run_lfpr import ce_gen, channels
from uw.run_gm import auc as rank_auc

N_FT = 24
SEED0 = 47901
F_SEED = 47950
THETA0 = ('research_outputs/uw/ew3-20260925T113900/ckpt/theta0.pt')


def rank_ci(fm, bm, fe, be, n=2000, seed=53900):
    rs = np.random.RandomState(seed)
    base, fin = rank_auc(bm, be), rank_auc(fm, fe)
    diffs = [rank_auc(fm[rs.choice(len(fm), len(fm), replace=True)],
                      fe) - rank_auc(bm[rs.choice(len(fm), len(fm),
                                                  replace=True)], be)
            for _ in range(n)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return base, fin, float(lo), float(hi)


def rec(src, idx, device):
    x = torch.tensor(np.stack([src.cifar10.data[i] / 255.0
                               for i in idx]))\
        .permute(0, 3, 1, 2).float().to(device)
    y = torch.tensor([src.cifar10.targets[i] for i in idx])\
        .long().to(device)
    return x, y


def run(cli):
    device = torch.device('cuda:{}'.format(cli.gpu))
    spec = dict(dataset='CIFAR10', arch='ResNet18', nc=10, cls=3,
                pool_scheme='build_pool', k=K)
    ctx = ez.make_context(spec, device)
    model = ctx['model']
    from tools.cifar_10 import CIFAR10
    import torchvision.transforms as _tf
    tset = CIFAR10(train=False, transform=_tf.Compose([_tf.ToTensor()]))
    cat_te = np.where(np.asarray(tset.cifar10.targets) == 3)[0]
    cat_tr = ctx['pool']

    # ---- random F instead of f_deep ------------------------------
    frand = np.sort(np.random.RandomState(F_SEED).choice(
        cat_tr, K, replace=False))
    keep = np.setdiff1d(np.arange(ctx['n']), frand).astype(int)
    trset = CIFAR10(train=True, transform=_tf.Compose([_tf.ToTensor()]))
    mem = rec(trset, frand, device)

    cons = np.sort(np.random.RandomState(52801).choice(cat_te, 300,
                                                       replace=False))
    rest = np.setdiff1d(cat_te, cons)
    fresh = np.sort(np.random.RandomState(52501).choice(rest, 200,
                                                        replace=False))
    fx = rec(tset, fresh, device)
    cx = rec(tset, cons, device)
    theta0 = torch.load(THETA0, map_location=device)

    ck = os.path.join(cli.outdir, 'ckpt')
    os.makedirs(ck, exist_ok=True)

    def train_or_load(s):
        p = os.path.join(ck, 'ft_s%d.pt' % s)
        if os.path.exists(p):
            return torch.load(p, map_location=device)
        sd = ctx['fit_full'](keep, s, 1, init=theta0, cosine=False)
        torch.save(sd, p)
        print('  randF ft seed %d done' % s, flush=True)
        return sd

    seeds = [SEED0 + i for i in range(N_FT)]
    sds = [train_or_load(s) for s in seeds]

    def mvec(sd):
        return ctx['eval_full'](sd)

    fit_mm = [mvec(sd) for sd in sds[:6]]
    c = {m: float(np.median([e[m] for e in fit_mm])) for m in METRICS}
    sig = {m: float(np.median(np.abs(
        np.array([e[m] for e in fit_mm]) - c[m]))) for m in METRICS}

    def sc(sd):
        e = mvec(sd)
        return max(abs(e[m] - c[m]) / max(sig[m], 1e-12)
                   for m in METRICS)

    cal_scores = [sc(sd) for sd in sds[6:18]]
    q = conformal_q(cal_scores, 0.20)
    hon_pass = sum(1 for sd in sds if float(sc(sd)) <= q)
    print('randF A6: q=%.3f honest %d/%d' % (q, hon_pass, N_FT),
          flush=True)

    out = {'arm': 'D-randF', 'f_seed': F_SEED, 'q': q,
           'honest_pass': '%d/%d' % (hon_pass, N_FT), 'rows': []}
    for se, sd0 in zip(seeds[18:], sds[18:]):
        s0 = float(sc(sd0))
        bm = channels(model, sd0, *mem)['neggrad']
        fe = channels(model, sd0, fx[0], fx[1])['neggrad']
        fin = ce_gen(model, mem, cx, sc, q, {'sd': sd0, 'n': 60})
        sf = float(sc(fin))
        fm = channels(model, fin, *mem)['neggrad']
        ch_fin_f = channels(model, fin, fx[0], fx[1])
        base_auc, fin_auc, lo, hi = rank_ci(
            fm, bm, ch_fin_f['neggrad'], fe)
        torch.save(fin, os.path.join(ck, 'enc_s%d.pt' % se))
        row = {'start_seed': se, 's6_start': s0, 's6_fin': sf,
               'pass_fin': bool(sf <= q),
               'auc_base': base_auc, 'auc_fin': fin_auc,
               'ci': [lo, hi], 'ci_positive': bool(lo > 0)}
        out['rows'].append(row)
        print('randF s%d: s6 %.3f->%.3f pass=%s auc %.3f->%.3f '
              'CI[%.3f,%.3f]'
              % (se, s0, sf, row['pass_fin'], base_auc, fin_auc,
                 lo, hi), flush=True)

    surv = sum(1 for r in out['rows'] if r['pass_fin'] and r['ci_positive'])
    out['n_survive'] = surv
    out['verdict'] = ('RANDF-REPLICATED' if surv >= 2
                      else 'RANDF-WEAK' if surv == 1
                      else 'RANDF-NOT')
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('VERDICT', out['verdict'], '(%d/3)' % surv, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default=1, type=int)
    ap.add_argument('--outdir', default='')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/abld-%s' % (
            time.strftime('%Y%m%dT%H%M%S'),)
    assert not os.path.exists(cli.outdir), cli.outdir
    os.makedirs(cli.outdir)
    run(cli)


if __name__ == '__main__':
    main()
